#ifndef TOKENINDEX_DEFINED
#define TOKENINDEX_DEFINED
typedef struct {
  const char *str;
  int len;
  int id;
} TokenIndex;
#endif // TOKENINDEX_DEFINED
/* Minimal GPT-OSS inference (single-node, CPU, pure C)
 *
 * Features implemented:
 *  - GQA attention (n_heads, n_kv_heads)
 *  - Alternating banded (sliding-window) and dense attention layers
 *  - RoPE with configurable scaling (placeholder for YaRN)
 *  - MoE FFN with top-K routing (default K=4)
 *  - Optional projection biases (GPT-OSS uses biases)
 *  - Tokenizer loader & BPE-like encode/"safe" decode (Karpathy
 * llama2.c-compatible tokenizer.bin)
 *  - Temperature + top-p sampling
 *
 * Build:
 *   cc -O3 -march=native -o gpt-oss gpt-oss.c -lm
 *
 * Run (after packing weights & exporting tokenizer.bin):
 *   ./gpt-oss gpt-oss-20b.bin -i "Hello, world" -n 64
 *
 * TODOs:
 *  - MXFP4 expert weights: decode on load or preconvert to fp16/fp32.
 *  - Fast paths: precomputed sin/cos for RoPE, vectorization, kv cache layout.
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

// -------------------------------
// Model config (matches checkpoint header)

typedef struct {
  int dim;          // residual dim
  int hidden_dim;   // FFN hidden dim per expert
  int n_layers;     // transformer layers
  int n_heads;      // query heads
  int n_kv_heads;   // KV heads (GQA)
  int n_experts;    // experts per layer (e.g., 128 for 120B, 32 for 20B)
  int top_k;        // router top-k (default 4)
  int vocab_size;   // tokenizer vocab (o200k_harmony = ~201088); if <0 => tied
  int seq_len;      // max sequence length supported by checkpoint
  int window;       // sliding window size for banded attention (e.g., 128)
  int alt_banded;   // if 1, layers alternate banded/dense starting at 0
  float rope_base;  // RoPE base theta (default 10000.0f)
  float rope_scale; // RoPE scaling (placeholder for YaRN)
} GPTOSSConfig;

// -------------------------------
// Weights layout

typedef struct {
  // token embedding
  float *tok_embeddings; // (vocab_size, dim)

  // per-layer RMSNorm weights
  float *rms_att_w; // (layer, dim)
  float *rms_ffn_w; // (layer, dim)

  // attention projections (with biases)
  float *wq; // (layer, dim, dim)
  float *wk; // (layer, dim, kv_dim)
  float *wv; // (layer, dim, kv_dim)
  float *wo; // (layer, dim, dim)  [note: stored as (out, in)]
  float *bq; // (layer, dim)
  float *bk; // (layer, kv_dim)
  float *bv; // (layer, kv_dim)

  float *bo; // (layer, dim)

  // learned attention sink bias (per layer, per head): added to softmax denom
  float *attn_sink; // (layer, n_heads)

  // MoE router (linear + bias): logits over experts
  float *wr; // (layer, n_experts, dim)
  float *br; // (layer, n_experts)

  // MoE experts: per-expert up/gate/down (SwiGLU variant)
  // W_up: (hidden, dim), W_gate: (hidden, dim), W_down: (dim, hidden)
  float *w_up;   // (layer, expert, hidden, dim)
  float *w_gate; // (layer, expert, hidden, dim)
  float *w_down; // (layer, expert, dim, hidden)
  float *b_up;   // (layer, expert, hidden)
  float *b_gate; // (layer, expert, hidden)
  float *b_down; // (layer, expert, dim)

  // final RMSNorm and LM head
  float *rms_final_w; // (dim)
  float *wcls;        // (vocab_size, dim) or shared with embeddings
} GPTOSSWeights;

// -------------------------------
// Runtime buffers

typedef struct {
  float *x;   // (dim)
  float *xb;  // (dim)
  float *xb2; // (dim)

  float *q; // (dim)
  float *k; // (kv_dim)
  float *v; // (kv_dim)

  float *hb;  // (hidden_dim)
  float *hb2; // (hidden_dim)

  float *att;    // (n_heads * seq_len)
  float *logits; // (vocab_size)

  // KV cache: (layer, seq_len, kv_dim)
  float *key_cache;
  float *val_cache;

  // router scratch
  float *router;      // (n_experts)
  int *topk_idx;      // (top_k)
  float *topk_weight; // (top_k) softmax weights
} RunState;

// -------------------------------
// Full model holder

typedef struct {
  GPTOSSConfig config;
  GPTOSSWeights weights;
  RunState state;
  // mmap housekeeping
  int fd;
  float *data;
  ssize_t file_size;
} GPTOSSModel;

// -------------------------------
// ==============================
// Tokenizer: robust encode/decode
// ==============================

#ifndef TOKENINDEX_DEFINED
#define TOKENINDEX_DEFINED
typedef struct {
  const char *str;
  int len;
  int id;
} TokenIndex;
#endif // TOKENINDEX_DEFINED

typedef struct {
  char *
      *vocab; // [vocab_size] UTF-8 strings; may contain raw 1-byte or "<0xHH>"
  int *lengths;  // [vocab_size] cached lengths
  float *scores; // [vocab_size] rank-like; lower (more negative) = earlier
                 // merge; specials very negative (~-1e30)
  int vocab_size;
  int max_token_length;

  TokenIndex *sorted_vocab; // for bsearch by (bytes,len)

  // Byte token maps (supports either style)
  // For each 0..255, gives token id or -1 if not present.
  int byte_tokens[256];

  // small decode scratch for "<0xHH>" path
  unsigned char byte_pieces[512]; // interleaved byte + NUL
} Tokenizer;

// --------- helpers

static int compare_tokens(const void *a, const void *b) {
  const TokenIndex *pa = (const TokenIndex *)a;
  const TokenIndex *pb = (const TokenIndex *)b;
  int minlen = pa->len < pb->len ? pa->len : pb->len;
  int cmp = memcmp(pa->str, pb->str, minlen);
  if (cmp != 0)
    return cmp;
  return pa->len - pb->len;
}

static int find_token_id(Tokenizer *t, const char *s, int len) {
  TokenIndex key = {.str = s, .len = len, .id = -1};
  TokenIndex *res = (TokenIndex *)bsearch(&key, t->sorted_vocab, t->vocab_size,
                                          sizeof(TokenIndex), compare_tokens);
  return res ? res->id : -1;
}

// C-safe helper to parse a single hex nibble
static inline int hex_nibble(int c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

// Detect "<0xHH>" format and parse to byte (returns 1 on success)
static int parse_hex_byte_token(const char *s, int len, unsigned char *out) {
  if (len != 6)
    return 0; // expects exactly "<0xHH>"
  if (s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>')
    return 0;

  int h = hex_nibble((unsigned char)s[3]);
  int l = hex_nibble((unsigned char)s[4]);
  if (h < 0 || l < 0)
    return 0;

  *out = (unsigned char)((h << 4) | l);
  return 1;
}

static void read_tokenizer(Tokenizer *t, const char *path, int vocab_size) {
  memset(t, 0, sizeof(*t));
  t->vocab_size = vocab_size;

  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "tokenizer open %s failed: %s\n", path, strerror(errno));
    exit(1);
  }

  if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) {
    fprintf(stderr, "bad tokenizer header\n");
    exit(1);
  }

  t->vocab = (char **)calloc(vocab_size, sizeof(char *));
  t->lengths = (int *)calloc(vocab_size, sizeof(int));
  t->scores = (float *)calloc(vocab_size, sizeof(float));
  t->sorted_vocab = (TokenIndex *)calloc(vocab_size, sizeof(TokenIndex));
  if (!t->vocab || !t->scores || !t->sorted_vocab || !t->lengths) {
    fprintf(stderr, "OOM tokenizer\n");
    exit(1);
  }

  for (int i = 0; i < vocab_size; i++) {
    float score;
    int len;
    if (fread(&score, sizeof(float), 1, f) != 1) {
      fprintf(stderr, "bad tokenizer file (score)\n");
      exit(1);
    }
    if (fread(&len, sizeof(int), 1, f) != 1) {
      fprintf(stderr, "bad tokenizer file (len)\n");
      exit(1);
    }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
      fprintf(stderr, "OOM\n");
      exit(1);
    }
    if ((int)fread(buf, 1, len, f) != len) {
      fprintf(stderr, "bad tokenizer file (str)\n");
      exit(1);
    }
    buf[len] = '\0';

    t->vocab[i] = buf;
    t->lengths[i] = len;
    t->scores[i] = score;
    t->sorted_vocab[i].str = buf;
    t->sorted_vocab[i].len = len;
    t->sorted_vocab[i].id = i;
  }
  fclose(f);

  qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);

  // find byte tokens either as "<0xHH>" or as single literal char
  char tmp[8];
  for (int b = 0; b < 256; b++) {
    int id = -1;
    // textual placeholder
    snprintf(tmp, sizeof(tmp), "<0x%02X>", b);
    id = find_token_id(t, tmp, (int)strlen(tmp));
    // literal single-byte (printable ASCII or common whitespace)
    if (id < 0) {
      char one[2] = {(char)b, 0};
      if (b == 9 || b == 10 || b == 13 || (b >= 32 && b <= 126)) {
        id = find_token_id(t, one, 1);
      }
    }
    t->byte_tokens[b] = id;
    // also prepare raw single-byte decode
    t->byte_pieces[b * 2] = (unsigned char)b;
    t->byte_pieces[b * 2 + 1] = '\0';
  }
}

static void free_tokenizer(Tokenizer *t) {
  if (!t)
    return;
  if (t->vocab) {
    for (int i = 0; i < t->vocab_size; i++)
      free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->lengths);
  free(t->scores);
  free(t->sorted_vocab);
}

// Greedy BPE with minimal allocations.
// - Start from raw bytes -> byte tokens
// - Repeatedly merge adjacent pairs if the concatenation exists
//   and is not "special" (score <= -1e29 treated as special marker).
static void encode(Tokenizer *t, const char *text, int bos_id, int eos_id,
                   int *out, int *n_out, int max_tokens) {
  const unsigned char *p = (const unsigned char *)text;
  int text_len = (int)strlen((const char *)p);

  // Worst-case capacity (bytes + BOS + EOS)
  int cap = text_len + 2;
  int *tokens = (int *)malloc(sizeof(int) * (size_t)cap);
  if (!tokens) {
    fprintf(stderr, "encode OOM\n");
    exit(1);
  }

  int ntok = 0;
  if (bos_id >= 0)
    tokens[ntok++] = bos_id;

  // 1) Bytes -> initial tokens
  for (int i = 0; i < text_len; i++) {
    int id = t->byte_tokens[p[i]];
    if (id < 0) {
      // Should not happen for a sane tokenizer; fallback to first vocab id
      // (usually <unk>)
      id = 0;
    }
    tokens[ntok++] = id;
  }

  // Scratch for candidate concatenations, allocated once up to max_token_length
  char *cat = (char *)malloc((size_t)t->max_token_length + 1);
  if (!cat) {
    fprintf(stderr, "encode OOM (cat)\n");
    exit(1);
  }

  // 2) Greedy merges: lowest score (rank) wins, skip specials
  // We'll maintain a parallel array of (piece ptr, len) to avoid strlen.
  const char **pieces = (const char **)malloc(sizeof(char *) * (size_t)ntok);
  int *plens = (int *)malloc(sizeof(int) * (size_t)ntok);
  if (!pieces || !plens) {
    fprintf(stderr, "encode OOM (pieces)\n");
    exit(1);
  }
  for (int i = 0; i < ntok; i++) {
    pieces[i] = t->vocab[tokens[i]];
    plens[i] = t->lengths[tokens[i]];
  }

  while (1) {
    float best_score = 1e29f;
    int best_idx = -1, best_id = -1;

    for (int i = 0; i < ntok - 1; i++) {
      int la = plens[i], lb = plens[i + 1];
      int lsum = la + lb;
      if (lsum > t->max_token_length)
        continue;

      // Build concatenation in scratch
      memcpy(cat, pieces[i], (size_t)la);
      memcpy(cat + la, pieces[i + 1], (size_t)lb);
      cat[lsum] = '\0';

      int id = find_token_id(t, cat, lsum);
      if (id < 0)
        continue;

      float score = t->scores[id];
      // Treat very negative scores as "special" => non-mergeable
      if (score <= -1e29f)
        continue;

      // Lower score = earlier merge (rank). Choose the minimum.
      if (score < best_score) {
        best_score = score;
        best_idx = i;
        best_id = id;
      }
    }

    if (best_idx < 0)
      break;

    // Commit the best merge: replace pair (best_idx, best_idx+1) with best_id
    tokens[best_idx] = best_id;
    pieces[best_idx] = t->vocab[best_id];
    plens[best_idx] = t->lengths[best_id];
    for (int j = best_idx + 1; j < ntok - 1; j++) {
      tokens[j] = tokens[j + 1];
      pieces[j] = pieces[j + 1];
      plens[j] = plens[j + 1];
    }
    ntok--;
  }

  // 3) Append EOS if requested and space remains
  if (eos_id >= 0 && ntok < cap)
    tokens[ntok++] = eos_id;

  // 4) Truncate if caller's buffer is smaller
  if (ntok > max_tokens)
    ntok = max_tokens;
  for (int i = 0; i < ntok; i++)
    out[i] = tokens[i];
  *n_out = ntok;

  free(tokens);
  free(pieces);
  free(plens);
  free(cat);
}

static char *decode_piece(Tokenizer *t, int prev_token, int token) {
  (void)prev_token;
  if (token < 0 || token >= t->vocab_size)
    return "";

  const char *piece = t->vocab[token];
  int len = t->lengths[token];

  // Byte token if:
  //  - single raw byte, OR
  //  - "<0xHH>" formatted
  if (len == 1) {
    // Return as a 1-byte C string
    static char out[2];
    out[0] = piece[0];
    out[1] = '\0';
    return out;
  }

  unsigned char b;
  if (parse_hex_byte_token(piece, len, &b)) {
    // Use per-byte scratch for reentrancy across different bytes
    return (char *)&t->byte_pieces[(int)b * 2];
  }

  // Normal string piece (UTF-8 safe as-is)
  return (char *)piece;
}

static void safe_printf(const char *s) {
  if (!s || !*s)
    return;
  // The pieces we output are proper NUL-terminated byte strings.
  // This prints raw bytes without sanitization (emoji/UTF-8 okay).
  fputs(s, stdout);
}

// -------------------------------
// Math helpers

static inline void rmsnorm(float *o, const float *x, const float *w, int size) {
  float ss = 0.0f;
  for (int i = 0; i < size; i++)
    ss += x[i] * x[i];
  ss = 1.0f / sqrtf(ss / size + 1e-5f);
  for (int i = 0; i < size; i++)
    o[i] = w[i] * (ss * x[i]);
}

static inline void softmax_inplace(float *x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; i++)
    if (x[i] > mx)
      mx = x[i];
  float s = 0.0f;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - mx);
    s += x[i];
  }
  float inv = 1.0f / s;
  for (int i = 0; i < n; i++)
    x[i] *= inv;
}

static inline void matmul(float *y, const float *x, const float *w, int n,
                          int d) {
  // y(d) = W(d,n) @ x(n)
  for (int i = 0; i < d; i++) {
    float acc = 0.0f;
    const float *wrow = w + (size_t)i * n;
    for (int j = 0; j < n; j++)
      acc += wrow[j] * x[j];
    y[i] = acc;
  }
}

// top-k indices by value (k <= n_experts). O(n*logk) partial insertion sort.
static inline void topk_indices(const float *x, int n, int k, int *idx,
                                float *val) {
  for (int i = 0; i < k; i++) {
    idx[i] = -1;
    val[i] = -1e30f;
  }
  for (int i = 0; i < n; i++) {
    float v = x[i];
    int pos = -1;
    for (int j = 0; j < k; j++)
      if (v > val[j]) {
        pos = j;
        break;
      }
    if (pos >= 0) {
      for (int j = k - 1; j > pos; j--) {
        val[j] = val[j - 1];
        idx[j] = idx[j - 1];
      }
      val[pos] = v;
      idx[pos] = i;
    }
  }
}

// -------------------------------
// Memory management

static void malloc_run_state(RunState *s, const GPTOSSConfig *p) {
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = (float *)calloc(p->dim, sizeof(float));
  s->xb = (float *)calloc(p->dim, sizeof(float));
  s->xb2 = (float *)calloc(p->dim, sizeof(float));
  s->q = (float *)calloc(p->dim, sizeof(float));
  s->k = (float *)calloc(kv_dim, sizeof(float));
  s->v = (float *)calloc(kv_dim, sizeof(float));
  s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
  s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
  s->att = (float *)calloc((size_t)p->n_heads * p->seq_len, sizeof(float));
  s->logits = (float *)calloc(p->vocab_size, sizeof(float));
  s->key_cache =
      (float *)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->val_cache =
      (float *)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->router =
      (float *)calloc(p->n_experts > 0 ? p->n_experts : 1, sizeof(float));
  s->topk_idx = (int *)calloc(p->top_k > 0 ? p->top_k : 1, sizeof(int));
  s->topk_weight = (float *)calloc(p->top_k > 0 ? p->top_k : 1, sizeof(float));
  if (!s->x || !s->xb || !s->xb2 || !s->q || !s->k || !s->v || !s->hb ||
      !s->hb2 || !s->att || !s->logits || !s->key_cache || !s->val_cache ||
      !s->router || !s->topk_idx || !s->topk_weight) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
}

static void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->q);
  free(s->k);
  free(s->v);
  free(s->hb);
  free(s->hb2);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->val_cache);
  free(s->router);
  free(s->topk_idx);
  free(s->topk_weight);
}

// Map weights from a single contiguous float* buffer. See README for pack
// order.
static void memory_map_weights(GPTOSSWeights *w, const GPTOSSConfig *p,
                               float *ptr, int shared_weights) {
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  size_t L = (size_t)p->n_layers;
  size_t E = (size_t)p->n_experts;

  w->tok_embeddings = ptr;
  ptr += (size_t)p->vocab_size * p->dim;
  w->rms_att_w = ptr;
  ptr += L * p->dim;
  w->rms_ffn_w = ptr;
  ptr += L * p->dim;

  w->wq = ptr;
  ptr += L * (size_t)p->dim * p->dim;
  w->wk = ptr;
  ptr += L * (size_t)p->dim * kv_dim;
  w->wv = ptr;
  ptr += L * (size_t)p->dim * kv_dim;
  w->wo = ptr;
  ptr += L * (size_t)p->dim * p->dim;
  w->bq = ptr;
  ptr += L * p->dim;
  w->bk = ptr;
  ptr += L * kv_dim;
  w->bv = ptr;
  ptr += L * kv_dim;

  w->bo = ptr;
  ptr += L * p->dim;

  // attention sink logits
  w->attn_sink = ptr;
  ptr += L * p->n_heads;

  if (p->n_experts > 0) {
    w->wr = ptr;
    ptr += L * E * p->dim;
    w->br = ptr;
    ptr += L * E;

    w->w_up = ptr;
    ptr += L * E * (size_t)p->hidden_dim * p->dim;
    w->w_gate = ptr;
    ptr += L * E * (size_t)p->hidden_dim * p->dim;
    w->w_down = ptr;
    ptr += L * E * (size_t)p->dim * p->hidden_dim;
    w->b_up = ptr;
    ptr += L * E * p->hidden_dim;
    w->b_gate = ptr;
    ptr += L * E * p->hidden_dim;
    w->b_down = ptr;
    ptr += L * E * p->dim;
  } else {
    w->wr = w->br = w->w_up = w->w_gate = w->w_down = w->b_up = w->b_gate =
        w->b_down = NULL;
  }

  w->rms_final_w = ptr;
  ptr += p->dim;
  w->wcls = shared_weights ? w->tok_embeddings : ptr;
}

// Checkpoint reader (simple mmap). Header must be sizeof(GPTOSSConfig).
static void read_checkpoint(const char *path, GPTOSSConfig *cfg,
                            GPTOSSWeights *w, int *fd, float **data,
                            ssize_t *file_size) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "open %s failed\n", path);
    exit(1);
  }
  if (fread(cfg, sizeof(GPTOSSConfig), 1, f) != 1) {
    fprintf(stderr, "bad header\n");
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  *file_size = ftell(f);
  fclose(f);
  *fd = open(path, O_RDONLY);
  if (*fd == -1) {
    perror("open");
    exit(1);
  }
  *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  int shared = (cfg->vocab_size < 0) ? 1 : 0; // NEGATIVE => tied
  cfg->vocab_size = abs(cfg->vocab_size);
  float *ptr = *data + sizeof(GPTOSSConfig) / sizeof(float);
  memory_map_weights(w, cfg, ptr, shared);
}

static void build_model(GPTOSSModel *m, const char *checkpoint) {
  read_checkpoint(checkpoint, &m->config, &m->weights, &m->fd, &m->data,
                  &m->file_size);
  malloc_run_state(&m->state, &m->config);
}

static void free_model(GPTOSSModel *m) {
  if (m->data && m->data != MAP_FAILED)
    munmap(m->data, m->file_size);
  if (m->fd != -1)
    close(m->fd);
  free_run_state(&m->state);
}

// -------------------------------
// YaRN-style NTK-by-parts scaling:
#define YARN_ALPHA 1.0f
#define YARN_BETA 32.0f
static inline float yarn_gamma(float r) {
  if (r < YARN_ALPHA)
    return 0.0f;
  if (r > YARN_BETA)
    return 1.0f;
  return (r - YARN_ALPHA) / (YARN_BETA - YARN_ALPHA);
}

static inline void apply_rope_yarn(float *vec, int size, int head_size, int pos,
                                   float rope_base, float s, int L_new) {
  if (s <= 0.0f)
    s = 1.0f;
  for (int i = 0; i < size; i += 2) {
    int j = (i % head_size) >> 1;
    float inv_freq = powf(rope_base, -(2.0f * (float)j) / (float)head_size);
    float L_orig = s * (float)L_new;
    float r = L_orig * inv_freq;
    float g = yarn_gamma(r);
    float m = s * (1.0f - g) + 1.0f * g;
    float angle = (float)pos * (m * inv_freq);
    float c = cosf(angle), sd = sinf(angle);
    float v0 = vec[i], v1 = vec[i + 1];
    vec[i] = v0 * c - v1 * sd;
    vec[i + 1] = v0 * sd + v1 * c;
  }
}

// -------------------------------
// Forward pass

static float *forward(GPTOSSModel *model, int token, int pos) {
  GPTOSSConfig *p = &model->config;
  GPTOSSWeights *w = &model->weights;
  RunState *s = &model->state;
  const int dim = p->dim;
  const int head_size = dim / p->n_heads;
  const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  const int kv_mul = p->n_heads / p->n_kv_heads;

  // token embedding
  memcpy(s->x, w->tok_embeddings + (size_t)token * dim, sizeof(float) * dim);

  for (int l = 0; l < p->n_layers; l++) {
    // --- Attention ---
    rmsnorm(s->xb, s->x, w->rms_att_w + (size_t)l * dim, dim);

    // qkv projections (with biases)
    matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim, dim, dim);
    matmul(s->k, s->xb, w->wk + (size_t)l * dim * kv_dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + (size_t)l * dim * kv_dim, dim, kv_dim);
    for (int i = 0; i < dim; i++)
      s->q[i] += w->bq[(size_t)l * dim + i];
    for (int i = 0; i < kv_dim; i++) {
      s->k[i] += w->bk[(size_t)l * kv_dim + i];
      s->v[i] += w->bv[(size_t)l * kv_dim + i];
    }

    // RoPE with YaRN scaling
    apply_rope_yarn(s->q, dim, head_size, pos, p->rope_base, p->rope_scale,
                    p->seq_len);
    apply_rope_yarn(s->k, kv_dim, head_size, pos, p->rope_base, p->rope_scale,
                    p->seq_len);

    // store k/v in cache for current pos
    size_t loff = (size_t)l * p->seq_len * kv_dim + (size_t)pos * kv_dim;
    memcpy(s->key_cache + loff, s->k, kv_dim * sizeof(float));
    memcpy(s->val_cache + loff, s->v, kv_dim * sizeof(float));

    // Attention window config
    int t_start = 0;
    if (p->alt_banded && p->window > 0 && (l % 2 == 0)) {
      int wstart = pos - (p->window - 1);
      if (wstart > 0)
        t_start = wstart;
    }

    // clear s->xb (attention output, packed by heads)
    memset(s->xb, 0, sizeof(float) * dim);

    for (int h = 0; h < p->n_heads; h++) {
      float *q = s->q + h * head_size;
      float *att = s->att + (size_t)h * p->seq_len;

      // learned sink logit for this head/layer (can be 0 if NULL)
      float sink = 0.0f;
      if (w->attn_sink)
        sink = w->attn_sink[(size_t)l * p->n_heads + h];

      // compute scores against cached keys in window [t_start..pos]
      for (int t = t_start; t <= pos; t++) {
        const float *krow =
            model->state.key_cache + (size_t)l * p->seq_len * kv_dim +
            (size_t)t * kv_dim + (size_t)(h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
          score += q[i] * krow[i];
        att[t] = score / sqrtf((float)head_size);
      }

      // softmax over the active window with sink in the denominator
      float mx = sink;
      for (int t = t_start; t <= pos; t++)
        if (att[t] > mx)
          mx = att[t];
      float sum = expf(sink - mx); // sink contribution only in denom
      for (int t = t_start; t <= pos; t++) {
        float e = expf(att[t] - mx);
        att[t] = e;
        sum += e;
      }
      float inv = 1.0f / sum;
      for (int t = t_start; t <= pos; t++)
        att[t] *= inv;

      // weighted sum of values
      float *out = s->xb + h * head_size;
      for (int t = t_start; t <= pos; t++) {
        const float a = att[t];
        const float *vrow =
            model->state.val_cache + (size_t)l * p->seq_len * kv_dim +
            (size_t)t * kv_dim + (size_t)(h / kv_mul) * head_size;
        for (int i = 0; i < head_size; i++)
          out[i] += a * vrow[i];
      }
    }

    // output projection + bias + residual
    matmul(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++)
      s->x[i] += s->xb2[i] + w->bo[(size_t)l * dim + i];

    // --- MoE FFN (top-k) ---
    rmsnorm(s->xb, s->x, w->rms_ffn_w + (size_t)l * dim, dim);

    if (p->n_experts > 0 && p->top_k > 0) {
      // router logits: (n_experts)
      matmul(s->router, s->xb, w->wr + (size_t)l * p->n_experts * dim, dim,
             p->n_experts);
      for (int e = 0; e < p->n_experts; e++)
        s->router[e] += w->br[(size_t)l * p->n_experts + e];

      // pick top-k experts and softmax their logits
      float topv[32]; // supports up to k<=32
      topk_indices(s->router, p->n_experts, p->top_k, s->topk_idx, topv);
      for (int i = 0; i < p->top_k; i++)
        s->topk_weight[i] = topv[i];
      softmax_inplace(s->topk_weight, p->top_k);

      // compute weighted sum of expert outputs
      memset(s->xb2, 0, sizeof(float) * dim);
      for (int r = 0; r < p->top_k; r++) {
        const int e = s->topk_idx[r];
        if (e < 0)
          continue;
        const float w_r = s->topk_weight[r];
        const size_t up_off =
            ((size_t)l * p->n_experts + e) * (size_t)p->hidden_dim * p->dim;
        const size_t gate_off = up_off; // same shape as up
        const size_t down_off =
            ((size_t)l * p->n_experts + e) * (size_t)p->dim * p->hidden_dim;
        const size_t bu_off =
            ((size_t)l * p->n_experts + e) * (size_t)p->hidden_dim;
        const size_t bg_off = bu_off;
        const size_t bd_off = ((size_t)l * p->n_experts + e) * (size_t)p->dim;

        // hb = W_up * x + b_up
        matmul(s->hb, s->xb, w->w_up + up_off, p->dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++)
          s->hb[i] += w->b_up[bu_off + i];
        // hb2 = W_gate * x + b_gate
        matmul(s->hb2, s->xb, w->w_gate + gate_off, p->dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++)
          s->hb2[i] += w->b_gate[bg_off + i];

        // SwiGLU (standard)
        for (int i = 0; i < p->hidden_dim; i++) {
          float v = s->hb[i];
          v *= (1.0f / (1.0f + expf(-v)));
          v *= s->hb2[i];
          s->hb[i] = v;
        }
        // down-projection + bias -> tmp into xb
        matmul(s->xb, s->hb, w->w_down + down_off, p->hidden_dim, p->dim);
        for (int i = 0; i < p->dim; i++)
          s->xb[i] += w->b_down[bd_off + i];

        // accumulate routed output
        for (int i = 0; i < dim; i++)
          s->xb2[i] += w_r * s->xb[i];
      }

      // residual
      for (int i = 0; i < dim; i++)
        s->x[i] += s->xb2[i];
    } else {
      // (Optional) dense FFN fallback could be implemented here if needed.
    }
  }

  // final rmsnorm + logits
  rmsnorm(s->x, s->x, w->rms_final_w, dim);
  matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
  return s->logits;
}

// -------------------------------
// Sampler

typedef struct {
  int vocab_size;
} Sampler;

static int sample_argmax(const float *p, int n) {
  int mi = 0;
  float mv = p[0];
  for (int i = 1; i < n; i++)
    if (p[i] > mv) {
      mv = p[i];
      mi = i;
    }
  return mi;
}

static int sample_next(Sampler *s, float *logits) {
  // Only greedy decoding: always pick argmax
  return sample_argmax(logits, s->vocab_size);
}

// -------------------------------
// Minimal CLI

#ifndef TESTING

static void usage() {
  fprintf(stderr, "Usage: gpt-oss <checkpoint.bin> -i \"prompt\" -n <steps> "
                  "[-t tokenizer.bin]\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc < 2)
    usage();
  const char *ckpt = argv[1];
  const char *prompt = "Hello";
  const char *tokpath = "tokenizer.bin";
  int steps = 64;
  // Only greedy decoding supported

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      prompt = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      steps = atoi(argv[++i]);
      continue;
    }
    if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
      tokpath = argv[++i];
      continue;
    }
    fprintf(stderr, "Unknown arg: %s\n", argv[i]);
    usage();
  }

  // Build model
  GPTOSSModel model = {0};
  build_model(&model, ckpt);
  if (steps <= 0 || steps > model.config.seq_len)
    steps = model.config.seq_len;

  // Load tokenizer
  Tokenizer tokenizer;
  read_tokenizer(&tokenizer, tokpath, model.config.vocab_size);

  // Discover special tokens in the vocab (o200k)
  int BOS = find_token_id(&tokenizer, "<|start|>", (int)strlen("<|start|>"));
  int EOS = find_token_id(&tokenizer, "<|return|>", (int)strlen("<|return|>"));
  // BOS = <|start|> (200006)
  // EOS = <|end|> (200007) for per-message, or <|return|> (200002) for
  // end-of-completion.
  printf("BOS token id: %d\n", BOS);
  printf("EOS token id: %d\n", EOS);

  // Encode prompt (only add BOS if we actually found it; don't auto-append EOS)
  int *tokens = (int *)malloc(sizeof(int) * (model.config.seq_len));
  int ntok = 0;
  encode(&tokenizer, prompt, BOS, -1, tokens, &ntok, model.config.seq_len);
  if (ntok < 1) {
    fprintf(stderr, "empty prompt after encoding\n");
    return 1;
  }

  // Sampler
  Sampler sampler = {.vocab_size = model.config.vocab_size};

  // ---- KV warmup: run the ENTIRE prompt through the model WITHOUT printing
  int pos = 0;
  int token = tokens[0];
  for (; pos < ntok - 1; pos++) {
    (void)forward(&model, token, pos);
    token = tokens[pos + 1];
  }
  // Now 'token' is the last prompt token and 'pos == ntok-1'.
  // We will start sampling from here and print ONLY new tokens.

  // Number of tokens to *generate* (not counting prompt)
  int to_generate = steps;
  if (to_generate > model.config.seq_len - ntok)
    to_generate = model.config.seq_len - ntok;

  // Generate continuation
  int generated = 0;
  while (generated < to_generate) {
    float *logits = forward(&model, token, pos);

    int next = sample_next(&sampler, logits);
    if (EOS >= 0 && next == EOS)
      break;

    char *piece = decode_piece(&tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);

    token = next;
    pos++;
    generated++;
  }
  printf("\n");

  free(tokens);
  free_tokenizer(&tokenizer);
  free_model(&model);
  return 0;
}
#endif

// gpt_oss_core.c
// Core loader, verifier, and forward pass for GPT-OSS.
// This file is self-contained; another file can #include it or link to it.

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

#pragma pack(push, 1)
typedef struct {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int n_experts;
  int top_k;
  int vocab_size; // NEGATIVE => tied output head
  int seq_len;
  int window;
  int alt_banded; // 0=dense, +1=even banded, -1=odd banded, +2=all banded
  float rope_base;
  float rope_scale;
} GPTOSSConfig;
#pragma pack(pop)

_Static_assert(sizeof(GPTOSSConfig) == 52,
               "GPTOSSConfig must be 52 bytes (<11i2f>)");

typedef struct {
  float *tok_embeddings; // [vocab, dim]
  float *rms_att_w;      // [L, dim]
  float *rms_ffn_w;      // [L, dim]
  float *wq;             // [L, dim, dim]
  float *wk;             // [L, kv_dim, dim]
  float *wv;             // [L, kv_dim, dim]
  float *wo;             // [L, dim, dim]
  float *bq;             // [L, dim]
  float *bk;             // [L, kv_dim]
  float *bv;             // [L, kv_dim]
  float *bo;             // [L, dim]
  float *attn_sink;      // [L, n_heads]
  // MoE
  float *wr;     // [L, E, dim]
  float *br;     // [L, E]
  float *w_up;   // [L, E, hidden, dim]
  float *w_gate; // [L, E, hidden, dim]
  float *w_down; // [L, E, dim, hidden]
  float *b_up;   // [L, E, hidden]
  float *b_gate; // [L, E, hidden]
  float *b_down; // [L, E, dim]
  // final
  float *rms_final_w; // [dim]
  float *wcls;        // [vocab, dim] or tied to tok_embeddings
} GPTOSSWeights;

typedef struct {
  float *x, *xb, *xb2;          // [dim]
  float *q;                     // [dim]
  float *k, *v;                 // [kv_dim]
  float *hb, *hb2;              // [hidden]
  float *att;                   // [n_heads * seq_len]
  float *logits;                // [vocab]
  float *key_cache, *val_cache; // [L, T, kv_dim]
  float *router;                // [E]
  int *topk_idx;                // [K]
  float *topk_weight;           // [K]
} RunState;

typedef struct {
  GPTOSSConfig cfg;
  GPTOSSWeights w;
  RunState s;
  int fd;
  void *mapped;
  size_t file_size;
} GPTOSSModel;

// ------------------------------ math helpers

static inline void rmsnorm(float *restrict o, const float *restrict x,
                           const float *restrict w, int size) {
  float ss = 0.0f;
  for (int j = 0; j < size; j++)
    ss += x[j] * x[j];
  float inv = 1.0f / sqrtf(ss / (float)size + 1e-5f);
  for (int j = 0; j < size; j++)
    o[j] = w[j] * (inv * x[j]);
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

// y[d] = W[d,n] @ x[n], W row-major (out, in)
static inline void matmul(float *restrict y, const float *restrict x,
                          const float *restrict w, int n, int d) {
  for (int i = 0; i < d; i++) {
    const float *wi = w + (size_t)i * n;
    float sum = 0.0f;
    for (int j = 0; j < n; j++)
      sum += wi[j] * x[j];
    y[i] = sum;
  }
}

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

// ------------------------------ allocation

static void malloc_run_state(RunState *s, const GPTOSSConfig *p) {
  const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

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

// ------------------------------ mapping

static void memory_map_weights(GPTOSSWeights *w, const GPTOSSConfig *p,
                               float *ptr, int shared_weights) {
  const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  const size_t L = (size_t)p->n_layers;
  const size_t E = (size_t)(p->n_experts > 0 ? p->n_experts : 1);

  w->tok_embeddings = ptr;
  ptr += (size_t)p->vocab_size * p->dim;

  w->rms_att_w = ptr;
  ptr += L * p->dim;
  w->rms_ffn_w = ptr;
  ptr += L * p->dim;

  w->wq = ptr;
  ptr += L * (size_t)p->dim * p->dim;
  w->wk = ptr;
  ptr += L * (size_t)kv_dim * p->dim;
  w->wv = ptr;
  ptr += L * (size_t)kv_dim * p->dim;
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
    w->wr = w->br = NULL;
    w->w_up = w->w_gate = w->w_down = NULL;
    w->b_up = w->b_gate = w->b_down = NULL;
  }

  w->rms_final_w = ptr;
  ptr += p->dim;
  w->wcls = shared_weights ? w->tok_embeddings : ptr;
}

// compute expected floats *excluding* lm_head if tied
static size_t expected_floats_min(const GPTOSSConfig *p) {
  const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  const size_t L = (size_t)p->n_layers;
  const size_t E = (size_t)(p->n_experts > 0 ? p->n_experts : 1);
  size_t tot = 0;
  tot += (size_t)p->vocab_size * p->dim;
  tot += L * p->dim * 2;                  // rms
  tot += L * (size_t)p->dim * p->dim * 2; // WQ + WO
  tot += L * (size_t)kv_dim * p->dim * 2; // WK + WV
  tot += L * (size_t)p->dim * 2;          // BQ + BO
  tot += L * (size_t)kv_dim * 2;          // BK + BV
  tot += L * p->n_heads;                  // sink
  if (p->n_experts > 0) {
    tot += L * E * p->dim;                             // WR
    tot += L * E;                                      // BR
    tot += L * E * (size_t)p->hidden_dim * p->dim * 2; // WUP/WGATE
    tot += L * E * (size_t)p->dim * p->hidden_dim;     // WDOWN
    tot += L * E * (size_t)p->hidden_dim * 2;          // BUP/BGATE
    tot += L * E * p->dim;                             // BDOWN
  }
  tot += p->dim; // rms_final
  return tot;
}

static int in_range(void *base, size_t total_bytes, const float *ptr,
                    size_t count_floats) {
  const uint8_t *b = (const uint8_t *)base;
  const uint8_t *p = (const uint8_t *)ptr;
  const uint8_t *e = p + count_floats * sizeof(float);
  return (p >= b) && (e <= (b + total_bytes)) && (p <= e);
}

static void section_stats(const char *name, const float *p, size_t n,
                          float *out_min, float *out_max, double *out_mean,
                          int *nan_count) {
  float mn = +INFINITY, mx = -INFINITY;
  double acc = 0.0;
  int nans = 0;
  for (size_t i = 0; i < n; i++) {
    float v = p[i];
    if (isnan(v)) {
      nans++;
      continue;
    }
    if (v < mn)
      mn = v;
    if (v > mx)
      mx = v;
    acc += (double)v;
  }
  if (out_min)
    *out_min = (nans == (int)n ? 0.0f : mn);
  if (out_max)
    *out_max = (nans == (int)n ? 0.0f : mx);
  if (out_mean)
    *out_mean = (nans == (int)n ? 0.0 : acc / (double)(n - nans));
  if (nan_count)
    *nan_count = nans;
  fprintf(stderr, "[VERIFY] %-12s  n=%zu  min=%g  max=%g  mean=%g  NaN=%d\n",
          name, n, (double)mn, (double)mx, (n ? acc / (double)(n - nans) : 0.0),
          nans);
}

static void verify_or_die(GPTOSSModel *m, int verbose) {
  GPTOSSConfig *p = &m->cfg;
  if (p->dim <= 0 || p->n_heads <= 0) {
    fprintf(stderr, "[ERR] bad dim/heads\n");
    exit(1);
  }
  if (p->dim % p->n_heads) {
    fprintf(stderr, "[ERR] dim %% n_heads != 0\n");
    exit(1);
  }
  if (p->n_kv_heads <= 0 || p->n_kv_heads > p->n_heads) {
    fprintf(stderr, "[ERR] bad n_kv_heads\n");
    exit(1);
  }
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  if (kv_dim * p->n_heads != p->dim * p->n_kv_heads) {
    fprintf(stderr, "[ERR] kv_dim mismatch\n");
    exit(1);
  }
  if (p->seq_len <= 0) {
    fprintf(stderr, "[ERR] seq_len <= 0\n");
    exit(1);
  }
  if (p->window < 0 || p->window > p->seq_len) {
    fprintf(stderr, "[ERR] bad window\n");
    exit(1);
  }

  size_t min_floats = expected_floats_min(p);
  size_t min_bytes = sizeof(GPTOSSConfig) + 4 * min_floats;
  if (m->file_size < min_bytes) {
    fprintf(stderr, "[ERR] file too small: %zu < %zu\n", m->file_size,
            min_bytes);
    exit(1);
  }
  // If LM head is untied, exact size should equal min_bytes + 4*vocab*dim
  size_t exact_with_head =
      min_bytes + 4ull * (size_t)p->vocab_size * (size_t)p->dim;
  if (m->file_size != min_bytes && m->file_size != exact_with_head) {
    fprintf(stderr, "[WARN] unexpected file size: %zu (expected %zu or %zu)\n",
            m->file_size, min_bytes, exact_with_head);
  }

// Range checks and quick stats for each section
#define CHK(ptr, count, name)                                                  \
  do {                                                                         \
    if (!(in_range(m->mapped, m->file_size, (ptr), (count)))) {                \
      fprintf(stderr, "[ERR] section %s out of range\n", (name));              \
      exit(1);                                                                 \
    }                                                                          \
    if (verbose) {                                                             \
      float mn, mx;                                                            \
      double mean;                                                             \
      int nans;                                                                \
      section_stats((name), (ptr), (count), &mn, &mx, &mean, &nans);           \
      if (isnan(mn) || isnan(mx) || nans)                                      \
        fprintf(stderr, "[WARN] NaNs in %s\n", (name));                        \
    }                                                                          \
  } while (0)

  const size_t L = (size_t)p->n_layers;
  const size_t E = (size_t)(p->n_experts > 0 ? p->n_experts : 1);

  CHK(m->w.tok_embeddings, (size_t)p->vocab_size * p->dim, "tok_embeddings");

  CHK(m->w.rms_att_w, L * p->dim, "rms_att_w");
  CHK(m->w.rms_ffn_w, L * p->dim, "rms_ffn_w");

  CHK(m->w.wq, L * (size_t)p->dim * p->dim, "wq");
  CHK(m->w.wk, L * (size_t)kv_dim * p->dim, "wk");
  CHK(m->w.wv, L * (size_t)kv_dim * p->dim, "wv");
  CHK(m->w.wo, L * (size_t)p->dim * p->dim, "wo");

  CHK(m->w.bq, L * p->dim, "bq");
  CHK(m->w.bk, L * kv_dim, "bk");
  CHK(m->w.bv, L * kv_dim, "bv");
  CHK(m->w.bo, L * p->dim, "bo");

  CHK(m->w.attn_sink, L * p->n_heads, "attn_sink");

  if (p->n_experts > 0) {
    CHK(m->w.wr, L * E * p->dim, "wr");
    CHK(m->w.br, L * E, "br");
    CHK(m->w.w_up, L * E * (size_t)p->hidden_dim * p->dim, "w_up");
    CHK(m->w.w_gate, L * E * (size_t)p->hidden_dim * p->dim, "w_gate");
    CHK(m->w.w_down, L * E * (size_t)p->dim * p->hidden_dim, "w_down");
    CHK(m->w.b_up, L * E * (size_t)p->hidden_dim, "b_up");
    CHK(m->w.b_gate, L * E * (size_t)p->hidden_dim, "b_gate");
    CHK(m->w.b_down, L * E * p->dim, "b_down");
  }

  CHK(m->w.rms_final_w, p->dim, "rms_final_w");

  if ((uint8_t *)m->w.wcls != (uint8_t *)m->w.tok_embeddings) {
    CHK(m->w.wcls, (size_t)p->vocab_size * p->dim, "wcls");
  }

#undef CHK

  fprintf(stderr,
          "[VERIFY] header: dim=%d hidden=%d L=%d H=%d KV=%d E=%d K=%d "
          "vocab=%d seq=%d win=%d alt=%d rope_base=%g rope_scale=%g\n",
          p->dim, p->hidden_dim, p->n_layers, p->n_heads, p->n_kv_heads,
          p->n_experts, p->top_k, p->vocab_size, p->seq_len, p->window,
          p->alt_banded, p->rope_base, p->rope_scale);
}

static void read_checkpoint_or_die(const char *path, GPTOSSModel *m) {
  memset(m, 0, sizeof(*m));
  m->fd = open(path, O_RDONLY);
  if (m->fd == -1) {
    perror("open");
    exit(1);
  }

  off_t fsz = lseek(m->fd, 0, SEEK_END);
  if (fsz <= 0) {
    fprintf(stderr, "bad file size\n");
    exit(1);
  }
  m->file_size = (size_t)fsz;
  if (lseek(m->fd, 0, SEEK_SET) < 0) {
    perror("lseek");
    exit(1);
  }

  m->mapped = mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
  if (m->mapped == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }

  // Copy header (packed)
  memcpy(&m->cfg, m->mapped, sizeof(GPTOSSConfig));
  if (!(m->cfg.rope_scale > 0.0f) || !isfinite(m->cfg.rope_scale))
    m->cfg.rope_scale = 1.0f;
  if (!(m->cfg.rope_base > 0.0f) || !isfinite(m->cfg.rope_base))
    m->cfg.rope_base = 10000.0f;

  int shared = (m->cfg.vocab_size < 0) ? 1 : 0;
  m->cfg.vocab_size = abs(m->cfg.vocab_size);

  float *ptr = (float *)((uint8_t *)m->mapped + sizeof(GPTOSSConfig));
  memory_map_weights(&m->w, &m->cfg, ptr, shared);
}

static void build_model_or_die(GPTOSSModel *m, const char *checkpoint,
                               int verify_verbose) {
  read_checkpoint_or_die(checkpoint, m);
  verify_or_die(m, verify_verbose);
  malloc_run_state(&m->s, &m->cfg);
}

static void free_model(GPTOSSModel *m) {
  if (m->mapped && m->mapped != MAP_FAILED)
    munmap(m->mapped, m->file_size);
  if (m->fd != -1)
    close(m->fd);
  free_run_state(&m->s);
}

// ------------------------------ RoPE (split-halves)

#define YARN_ALPHA 1.0f
#define YARN_BETA 32.0f
static inline float yarn_gamma(float r) {
  if (r <= YARN_ALPHA)
    return 0.0f;
  if (r >= YARN_BETA)
    return 1.0f;
  return (r - YARN_ALPHA) / (YARN_BETA - YARN_ALPHA);
}

static inline void apply_rope_yarn_halves(float *vec, int size, int head_size,
                                          int pos, float rope_base, float s,
                                          int L_new) {
  if (!(s > 0.0f))
    s = 1.0f;
  if (s > 1024.0f)
    s = 1024.0f;

  const int H = head_size;
  const int HH = H >> 1;
  const int n_heads_here = size / H;
  const float concentration = (s > 1.0f) ? (1.0f + 0.1f * logf(s)) : 1.0f;
  const float L_orig_scale = s * (float)L_new;

  for (int h = 0; h < n_heads_here; h++) {
    float *base = vec + h * H;
    for (int j = 0; j < HH; j++) {
      float inv_freq = powf(rope_base, -(2.0f * (float)j) / (float)H);
      float r = L_orig_scale * inv_freq;
      float g = yarn_gamma(r);
      float m = s * (1.0f - g) + 1.0f * g;
      float angle = (float)pos * (m * inv_freq);
      float c = cosf(angle), sd = sinf(angle);
      float v1 = base[j];
      float v2 = base[j + HH];
      base[j] = concentration * (v1 * c - v2 * sd);
      base[j + HH] = concentration * (v1 * sd + v2 * c);
    }
  }
}

// ------------------------------ forward (single token)

static float *forward_token(GPTOSSModel *m, int token, int pos) {
  GPTOSSConfig *p = &m->cfg;
  GPTOSSWeights *w = &m->w;
  RunState *s = &m->s;

  const int dim = p->dim;
  const int head_size = dim / p->n_heads;
  const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

  memcpy(s->x, w->tok_embeddings + (size_t)token * dim, sizeof(float) * dim);

  for (int l = 0; l < p->n_layers; l++) {
    rmsnorm(s->xb, s->x, w->rms_att_w + (size_t)l * dim, dim);

    matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim, dim, dim);
    matmul(s->k, s->xb, w->wk + (size_t)l * kv_dim * dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + (size_t)l * kv_dim * dim, dim, kv_dim);

    for (int i = 0; i < dim; i++)
      s->q[i] += w->bq[(size_t)l * dim + i];
    for (int i = 0; i < kv_dim; i++)
      s->k[i] += w->bk[(size_t)l * kv_dim + i];
    for (int i = 0; i < kv_dim; i++)
      s->v[i] += w->bv[(size_t)l * kv_dim + i];

    apply_rope_yarn_halves(s->q, dim, head_size, pos, p->rope_base,
                           p->rope_scale, p->seq_len);
    apply_rope_yarn_halves(s->k, kv_dim, head_size, pos, p->rope_base,
                           p->rope_scale, p->seq_len);

    const size_t loff = (size_t)l * p->seq_len * kv_dim + (size_t)pos * kv_dim;
    memcpy(s->key_cache + loff, s->k, kv_dim * sizeof(float));
    memcpy(s->val_cache + loff, s->v, kv_dim * sizeof(float));

    int apply_window = 0;
    if (p->window > 0) {
      if (p->alt_banded == 0)
        apply_window = 0;
      else if (p->alt_banded == 2)
        apply_window = 1;
      else if (p->alt_banded > 0)
        apply_window = ((l % 2) == 0);
      else
        apply_window = ((l % 2) == 1);
    }
    int t_start = 0;
    if (apply_window) {
      int wstart = pos - (p->window - 1);
      if (wstart > 0)
        t_start = wstart;
    }

    memset(s->xb, 0, (size_t)dim * sizeof(float));

    for (int h = 0; h < p->n_heads; h++) {
      float *q = s->q + (size_t)h * head_size;
      float *att = s->att + (size_t)h * p->seq_len;

      for (int t = t_start; t <= pos; t++) {
        const float *krow = s->key_cache + (size_t)l * p->seq_len * kv_dim +
                            (size_t)t * kv_dim +
                            (size_t)(h % p->n_kv_heads) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
          score += q[i] * krow[i];
        att[t] = score / sqrtf((float)head_size);
      }

      float mx = -1e30f;
      for (int t = t_start; t <= pos; t++)
        if (att[t] > mx)
          mx = att[t];
      float sum = 0.0f;
      for (int t = t_start; t <= pos; t++) {
        float e = expf(att[t] - mx);
        att[t] = e;
        sum += e;
      }
      float sink = w->attn_sink[(size_t)l * p->n_heads + h];
      sum += expf(sink - mx);
      float inv = 1.0f / sum;
      for (int t = t_start; t <= pos; t++)
        att[t] *= inv;

      float *out = s->xb + (size_t)h * head_size;
      for (int t = t_start; t <= pos; t++) {
        const float a = att[t];
        const float *vrow = s->val_cache + (size_t)l * p->seq_len * kv_dim +
                            (size_t)t * kv_dim +
                            (size_t)(h % p->n_kv_heads) * head_size;
        for (int i = 0; i < head_size; i++)
          out[i] += a * vrow[i];
      }
    }

    matmul(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++)
      s->x[i] += s->xb2[i] + w->bo[(size_t)l * dim + i];

    // MoE
    rmsnorm(s->xb, s->x, w->rms_ffn_w + (size_t)l * dim, dim);

    if (p->n_experts > 0 && p->top_k > 0) {
      matmul(s->router, s->xb, w->wr + (size_t)l * p->n_experts * dim, dim,
             p->n_experts);
      for (int e = 0; e < p->n_experts; e++)
        s->router[e] += w->br[(size_t)l * p->n_experts + e];

      float topv[256]; // enough
      topk_indices(s->router, p->n_experts, p->top_k, s->topk_idx, topv);
      for (int i = 0; i < p->top_k; i++)
        s->topk_weight[i] = topv[i];
      softmax_inplace(s->topk_weight, p->top_k);

      memset(s->xb2, 0, (size_t)dim * sizeof(float));
      const float alpha = 1.702f, limit = 7.0f;

      for (int r = 0; r < p->top_k; r++) {
        const int e = s->topk_idx[r];
        if (e < 0)
          continue;
        const float w_r = s->topk_weight[r];

        const size_t up_off =
            ((size_t)l * p->n_experts + e) * (size_t)p->hidden_dim * p->dim;
        const size_t gate_off = up_off;
        const size_t down_off =
            ((size_t)l * p->n_experts + e) * (size_t)p->dim * p->hidden_dim;

        const size_t bu_off =
            ((size_t)l * p->n_experts + e) * (size_t)p->hidden_dim;
        const size_t bg_off = bu_off;
        const size_t bd_off = ((size_t)l * p->n_experts + e) * (size_t)p->dim;

        matmul(s->hb, s->xb, w->w_up + up_off, p->dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++)
          s->hb[i] += w->b_up[bu_off + i];

        matmul(s->hb2, s->xb, w->w_gate + gate_off, p->dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++)
          s->hb2[i] += w->b_gate[bg_off + i];

        for (int i = 0; i < p->hidden_dim; i++) {
          float u = s->hb[i], g = s->hb2[i];
          if (u > limit)
            u = limit;
          if (g > limit)
            g = limit;
          if (g < -limit)
            g = -limit;
          float silu = u * (1.0f / (1.0f + expf(-alpha * u)));
          s->hb[i] = silu * (g + 1.0f);
        }

        matmul(s->xb, s->hb, w->w_down + down_off, p->hidden_dim, p->dim);
        for (int i = 0; i < p->dim; i++)
          s->xb[i] += w->b_down[bd_off + i];

        for (int i = 0; i < p->dim; i++)
          s->xb2[i] += w_r * s->xb[i];
      }

      for (int i = 0; i < p->dim; i++)
        s->x[i] += s->xb2[i];
    }
  }

  rmsnorm(s->x, s->x, w->rms_final_w, p->dim);
  matmul(s->logits, s->x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

// ------------------------------ public-ish API

// Build with verification (verbose: 0/1)
static void gptoss_build(GPTOSSModel *m, const char *bin_path,
                         int verify_verbose) {
  build_model_or_die(m, bin_path, verify_verbose);
}

static void gptoss_free(GPTOSSModel *m) { free_model(m); }

static float *gptoss_forward(GPTOSSModel *m, int token, int pos) {
  return forward_token(m, token, pos);
}

// gpt_oss_core.c — core loader + verifier + forward
// included by gpt_oss_main.c

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
// header written by exporter (11 ints + 2 floats = 52 bytes)
typedef struct {
  int dim;        // residual width
  int hidden_dim; // FFN hidden (per expert or dense)
  int n_layers;
  int n_heads;    // stored "sink heads" used for attn_sinks (not necessarily
                  // dim/head_dim)
  int n_kv_heads; // KV heads (GQA)
  int n_experts;  // 0 => dense
  int top_k;      // router top-k (unused if n_experts==0)
  int vocab_size; // NEGATIVE => tied LM head
  int seq_len;
  int window;
  int alt_banded; // 0 dense, ±1 alternate, 2 all banded
  float rope_base;
  float rope_scale;
} GPTOSSConfig;
#pragma pack(pop)
_Static_assert(sizeof(GPTOSSConfig) == 52, "header must be 52 bytes");

// ---------- UPDATED: weights, add MoE pointers ----------
typedef struct {
  float *tok_embeddings; // [vocab, dim]
  float *rms_att_w;      // [L, dim]
  float *rms_ffn_w;      // [L, dim]
  float *wq;             // [L, q_dim, dim]
  float *wk;             // [L, kv_dim, dim]
  float *wv;             // [L, kv_dim, dim]
  float *wo;             // [L, dim, q_dim]
  float *bq;             // [L, q_dim]
  float *bk;             // [L, kv_dim]
  float *bv;             // [L, kv_dim]
  float *bo;             // [L, dim]
  float *attn_sink;      // [L, sink_heads]
  // MoE (optional)
  float *wr;    // [L, E, dim]
  float *br;    // [L, E]
  float *wup;   // [L, E, hidden, dim]
  float *wgate; // [L, E, hidden, dim]
  float *wdown; // [L, E, dim, hidden]
  float *bup;   // [L, E, hidden]
  float *bgate; // [L, E, hidden]
  float *bdown; // [L, E, dim]

  float *rms_final_w; // [dim]
  float *wcls;        // [vocab, dim] (or tied)
} GPTOSSWeights;

typedef struct {
  // derived at load time
  int head_dim;   // per-head width
  int kv_dim;     // n_kv_heads * head_dim
  int q_dim;      // q_heads * head_dim
  int q_heads;    // q_dim / head_dim
  int sink_heads; // header.n_heads

  // activations/scratch
  float *x, *xb, *xb2;
  float *q, *k, *v, *xq;
  float *hb, *hb2; // (unused if dense FFN)
  float *att, *logits;
  float *key_cache, *val_cache;
  float *router;
  int *topk_idx;
  float *topk_weight; // (unused if dense FFN)
} RunState;

typedef struct {
  GPTOSSConfig cfg;
  GPTOSSWeights w;
  RunState s;

  // mmap bookkeeping
  int fd;
  void *mapped;
  size_t file_size;

  int tied; // 1 if header.vocab_size was negative (tied output head)
} GPTOSSModel;

// ---------- tiny math ----------
static inline void rmsnorm(float *o, const float *x, const float *w, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++)
    ss += x[i] * x[i];
  float inv = 1.0f / sqrtf(ss / (float)n + 1e-5f);
  for (int i = 0; i < n; i++)
    o[i] = w[i] * (inv * x[i]);
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
  // y(d) = W(d,n) @ x(n) ; W row-major by d
  for (int i = 0; i < d; i++) {
    const float *wi = w + (size_t)i * n;
    float s = 0.0f;
    for (int j = 0; j < n; j++)
      s += wi[j] * x[j];
    y[i] = s;
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
    for (int j = 0; j < k; j++) {
      if (v > val[j]) {
        pos = j;
        break;
      }
    }
    if (pos >= 0) {
      for (int j = k - 1; j > pos; j++) {
        val[j] = val[j - 1];
        idx[j] = idx[j - 1];
      }
      val[pos] = v;
      idx[pos] = i;
    }
  }
}

static int solve_dims_moe_aware(const GPTOSSConfig *p, size_t floats_after_hdr,
                                int tied, int *out_head_dim, int *out_kv_dim,
                                int *out_q_dim, int *out_q_heads,
                                int *out_sink_heads, int *out_experts) {
  const size_t L = (size_t)p->n_layers, D = (size_t)p->dim,
               V = (size_t)p->vocab_size;
  const int hidden = p->hidden_dim;

  // fixed parts that do NOT depend on q_dim/q_heads or MoE
  size_t prefix = V * D + L * D * 2; // embeddings + two RMS
  // KV-dependent but not q-dependent
  // we'll fill kv_dim from head_dim * n_kv_heads per candidate
  // tail pieces independent of q and MoE:
  size_t tail_fixed_base =
      L * (size_t)D /*BO*/ + D /*final RMS*/ + (tied ? 0 : V * D);

  // try common head_dims
  const int cand[] = {64, 80, 96, 128, 48, 40, 32};
  const int NC = (int)(sizeof(cand) / sizeof(cand[0]));

  for (int ci = 0; ci < NC; ++ci) {
    int hd = cand[ci];
    int kv_dim = hd * p->n_kv_heads;

    size_t kv_block =
        L * ((size_t)2 * kv_dim * D + (size_t)2 * kv_dim); // WK+WV + BK+BV

    if (floats_after_hdr < prefix + kv_block + tail_fixed_base)
      continue;

    // Remainder = [q parts + sinks + optional MoE]
    size_t R = floats_after_hdr - prefix - kv_block - tail_fixed_base;

    // write q_dim = hd * QH ; sinks use QH as well
    // q-dependent contribution: L*QH*((2*D+1)*hd + 1)
    const size_t a = (size_t)((2 * D + 1) * hd + 1);

    // MoE contribution per expert: L*(3*hidden*D + 2*D + 2*hidden + 1)
    const size_t b =
        (size_t)(3 * (size_t)hidden * D + 2 * D + 2 * (size_t)hidden + 1);

    if (R % L)
      continue; // must be multiple of L
    size_t R1 = R / L;

    // Scan plausible QH (query heads). Realistic range keeps it tiny.
    for (int QH = 1; QH <= 256; ++QH) {
      size_t q_term = (size_t)a * (size_t)QH;
      if (R1 < q_term)
        break; // too big already
      size_t rem = R1 - q_term;

      if (rem == 0) {
        // no MoE case
        *out_head_dim = hd;
        *out_kv_dim = kv_dim;
        *out_q_heads = QH;
        *out_q_dim = hd * QH;
        *out_sink_heads = QH;
        *out_experts = 0;
        return 1;
      }
      if (b > 0 && rem % b == 0) {
        int E = (int)(rem / b);
        if (E > 0) {
          *out_head_dim = hd;
          *out_kv_dim = kv_dim;
          *out_q_heads = QH;
          *out_q_dim = hd * QH;
          *out_sink_heads = QH; // sinks sized by query heads
          *out_experts = E;
          return 1;
        }
      }
    }
  }
  return 0;
}

// ---------- state alloc ----------
static void malloc_run_state(RunState *s, const GPTOSSConfig *p) {
  const int dim = p->dim, q_dim = s->q_dim, kv_dim = s->kv_dim;
  s->x = (float *)calloc(dim, sizeof(float));
  s->xb = (float *)calloc(dim, sizeof(float));
  s->xb2 = (float *)calloc(dim, sizeof(float));
  s->q = (float *)calloc(q_dim, sizeof(float));
  s->k = (float *)calloc(kv_dim, sizeof(float));
  s->v = (float *)calloc(kv_dim, sizeof(float));
  s->xq = (float *)calloc(q_dim, sizeof(float));
  s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
  s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
  s->att = (float *)calloc((size_t)s->q_heads * p->seq_len, sizeof(float));
  s->logits = (float *)calloc(p->vocab_size, sizeof(float));
  s->key_cache =
      (float *)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->val_cache =
      (float *)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->router =
      (float *)calloc(p->n_experts > 0 ? p->n_experts : 1, sizeof(float));
  s->topk_idx = (int *)calloc(p->top_k > 0 ? p->top_k : 1, sizeof(int));
  s->topk_weight = (float *)calloc(p->top_k > 0 ? p->top_k : 1, sizeof(float));
  if (!s->x || !s->xb || !s->xb2 || !s->q || !s->k || !s->v || !s->xq ||
      !s->hb || !s->hb2 || !s->att || !s->logits || !s->key_cache ||
      !s->val_cache || !s->router || !s->topk_idx || !s->topk_weight) {
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
  free(s->xq);
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

// ---------- mmap ----------
static void map_or_die(const char *path, GPTOSSModel *m) {
  memset(m, 0, sizeof(*m));
  m->fd = open(path, O_RDONLY);
  if (m->fd == -1) {
    perror("open");
    exit(1);
  }
  off_t sz = lseek(m->fd, 0, SEEK_END);
  if (sz <= 0) {
    fprintf(stderr, "bad file size\n");
    exit(1);
  }
  m->file_size = (size_t)sz;
  if (lseek(m->fd, 0, SEEK_SET) < 0) {
    perror("lseek");
    exit(1);
  }
  m->mapped = mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
  if (m->mapped == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }

  memcpy(&m->cfg, m->mapped, sizeof(GPTOSSConfig));
  m->tied = (m->cfg.vocab_size < 0);
  if (m->tied)
    m->cfg.vocab_size = -m->cfg.vocab_size;

  if (!(m->cfg.rope_scale > 0.0f) || !isfinite(m->cfg.rope_scale))
    m->cfg.rope_scale = 1.0f;
  if (!(m->cfg.rope_base > 0.0f) || !isfinite(m->cfg.rope_base))
    m->cfg.rope_base = 10000.0f;
}

// ---------- UPDATED: map with sinks-sized-by-q_heads + optional MoE ----------
static void memory_map_weights(GPTOSSWeights *w, const GPTOSSConfig *p,
                               float *ptr, int shared, int q_dim, int kv_dim,
                               int sink_heads, int n_experts) {
  size_t L = (size_t)p->n_layers, D = (size_t)p->dim, V = (size_t)p->vocab_size,
         H = (size_t)p->hidden_dim;

  w->tok_embeddings = ptr;
  ptr += V * D;

  w->rms_att_w = ptr;
  ptr += L * D;
  w->rms_ffn_w = ptr;
  ptr += L * D;

  w->wq = ptr;
  ptr += L * (size_t)q_dim * D;
  w->wk = ptr;
  ptr += L * (size_t)kv_dim * D;
  w->wv = ptr;
  ptr += L * (size_t)kv_dim * D;
  w->wo = ptr;
  ptr += L * D * (size_t)q_dim;

  w->bq = ptr;
  ptr += L * (size_t)q_dim;
  w->bk = ptr;
  ptr += L * (size_t)kv_dim;
  w->bv = ptr;
  ptr += L * (size_t)kv_dim;
  w->bo = ptr;
  ptr += L * D;

  w->attn_sink = ptr;
  ptr += L * (size_t)sink_heads;

  // Optional MoE block (8 parts)
  if (n_experts > 0) {
    w->wr = ptr;
    ptr += L * (size_t)n_experts * D;
    w->br = ptr;
    ptr += L * (size_t)n_experts;
    w->wup = ptr;
    ptr += L * (size_t)n_experts * H * D;
    w->wgate = ptr;
    ptr += L * (size_t)n_experts * H * D;
    w->wdown = ptr;
    ptr += L * (size_t)n_experts * D * H;
    w->bup = ptr;
    ptr += L * (size_t)n_experts * H;
    w->bgate = ptr;
    ptr += L * (size_t)n_experts * H;
    w->bdown = ptr;
    ptr += L * (size_t)n_experts * D;
  } else {
    w->wr = w->br = w->wup = w->wgate = w->wdown = w->bup = w->bgate =
        w->bdown = NULL;
  }

  w->rms_final_w = ptr;
  ptr += D;
  w->wcls = shared ? w->tok_embeddings : ptr;
}

// ---------- solve (head_dim, kv_dim, q_dim) by scanning head_dim ----------
static int solve_dims_from_size(const GPTOSSConfig *p, size_t floats_after_hdr,
                                int tied, int *out_head_dim, int *out_kv_dim,
                                int *out_q_dim, int *out_q_heads) {
  // fixed sections independent of q_dim/kv_dim
  const size_t L = (size_t)p->n_layers, D = (size_t)p->dim,
               V = (size_t)p->vocab_size;
  size_t prefix = V * D + L * D * 2;           // embeddings + two RMS
  size_t tail_fixed = L * (size_t)D            // BO
                      + L * (size_t)p->n_heads // attn_sinks
                      + D;                     // final RMS
  size_t lm_tail = tied ? 0 : V * D;

  // try common head_dim values (put 64 first; you can extend list if needed)
  const int cand[] = {64, 80, 96, 128, 48, 40, 32};
  const int NC = (int)(sizeof(cand) / sizeof(cand[0]));
  for (int ci = 0; ci < NC; ++ci) {
    int head_dim = cand[ci];
    int kv_dim = head_dim * p->n_kv_heads; // K,V rows
    // parts that depend on kv_dim but not on q_dim
    size_t kv_block =
        L * ((size_t)2 * kv_dim * D + (size_t)2 * kv_dim); // WK+WV and BK+BV
    // remainder should correspond to q parts: L*(2*q_dim*D + q_dim)
    if (floats_after_hdr < prefix + kv_block + tail_fixed + lm_tail)
      continue;
    size_t rem = floats_after_hdr - prefix - kv_block - tail_fixed - lm_tail;
    size_t coeff = L * ((size_t)2 * D + 1);
    if (coeff == 0 || rem % coeff)
      continue;
    size_t q_dim = rem / coeff;
    if (q_dim == 0 || (q_dim % (size_t)head_dim))
      continue;
    int q_heads = (int)(q_dim / (size_t)head_dim);
    if (q_heads <= 0)
      continue;

    // found a consistent solution
    *out_head_dim = head_dim;
    *out_kv_dim = kv_dim;
    *out_q_dim = (int)q_dim;
    *out_q_heads = q_heads;
    return 1;
  }
  return 0;
}

// ---------- build with verification ----------
// ---------- UPDATED: builder (verify + clamp seq_len) ----------
static void gptoss_build(GPTOSSModel *m, const char *bin_path, int verbose) {
  map_or_die(bin_path, m);
  GPTOSSConfig *p = &m->cfg;

  // sanity / clamps
  if (!(p->rope_scale > 0.0f) || !isfinite(p->rope_scale))
    p->rope_scale = 1.0f;
  if (!(p->rope_base > 0.0f) || !isfinite(p->rope_base))
    p->rope_base = 10000.0f;
  if (p->seq_len > 16384) {
    fprintf(stderr, "[WARN] clamping seq_len %d -> 4096\n", p->seq_len);
    p->seq_len = 4096;
  }

  const size_t floats_after_hdr =
      (m->file_size - sizeof(GPTOSSConfig)) / sizeof(float);

  int head_dim = -1, kv_dim = -1, q_dim = -1, q_heads = -1, sink_heads = -1,
      n_experts = -1;
  if (!solve_dims_moe_aware(p, floats_after_hdr, m->tied, &head_dim, &kv_dim,
                            &q_dim, &q_heads, &sink_heads, &n_experts)) {
    fprintf(stderr, "[ERR] could not infer dims/experts from file size. Export "
                    "layout mismatch.\n");
    exit(1);
  }

  // fill run-state & override obviously-wrong header fields
  m->s.head_dim = head_dim;
  m->s.kv_dim = kv_dim;
  m->s.q_dim = q_dim;
  m->s.q_heads = q_heads;
  m->s.sink_heads = sink_heads;
  if (n_experts >= 0)
    p->n_experts = n_experts; // trust the file, not the header

  if (verbose) {
    fprintf(stderr,
            "[VERIFY] header(raw): dim=%d hidden=%d L=%d n_heads(hdr)=%d "
            "n_kv=%d E(hdr)=%d top_k=%d vocab=%d seq=%d win=%d alt=%d "
            "rope_base=%.1f rope_scale=%.3f tied=%d\n",
            p->dim, p->hidden_dim, p->n_layers,
            ((GPTOSSConfig *)m->mapped)->n_heads, p->n_kv_heads, p->n_experts,
            p->top_k, p->vocab_size, p->seq_len, p->window, p->alt_banded,
            p->rope_base, p->rope_scale, m->tied);
    fprintf(stderr,
            "[VERIFY] derived: head_dim=%d kv_dim=%d q_dim=%d q_heads=%d "
            "sink_heads=%d experts=%d\n",
            head_dim, kv_dim, q_dim, q_heads, sink_heads, p->n_experts);
  }

  // map weights with derived sizes (sinks sized by q_heads)
  float *ptr = (float *)((uint8_t *)m->mapped + sizeof(GPTOSSConfig));
  memory_map_weights(&m->w, p, ptr, m->tied, q_dim, kv_dim, sink_heads,
                     p->n_experts);

  // final recount including optional MoE and sinks=q_heads
  const size_t L = (size_t)p->n_layers, D = (size_t)p->dim,
               V = (size_t)p->vocab_size, H = (size_t)p->hidden_dim;
  size_t expect = 0;
  expect += V * D;                      // embeddings
  expect += L * D + L * D;              // two RMS
  expect += L * (size_t)q_dim * D;      // WQ
  expect += L * (size_t)kv_dim * D * 2; // WK+WV
  expect += L * D * (size_t)q_dim;      // WO
  expect += L * (size_t)q_dim;          // BQ
  expect += L * (size_t)kv_dim * 2;     // BK+BV
  expect += L * D;                      // BO
  expect += L * (size_t)sink_heads;     // sinks
  if (p->n_experts > 0) {
    expect += L * (size_t)p->n_experts * D;         // WR
    expect += L * (size_t)p->n_experts;             // BR
    expect += L * (size_t)p->n_experts * H * D * 2; // WUP + WGATE
    expect += L * (size_t)p->n_experts * D * H;     // WDOWN
    expect += L * (size_t)p->n_experts * H * 2;     // BUP + BGATE
    expect += L * (size_t)p->n_experts * D;         // BDOWN
  }
  expect += D; // final RMS
  if (!m->tied)
    expect += V * D; // lm head

  if (expect != floats_after_hdr) {
    fprintf(stderr, "[ERR] recount mismatch: file=%zu floats, expect=%zu\n",
            floats_after_hdr, expect);
    exit(1);
  }

  malloc_run_state(&m->s, p);
}

// ---------- YaRN-style RoPE (same schedule you described) ----------
#define YARN_ALPHA 1.0f
#define YARN_BETA 32.0f
static inline float yarn_gamma(float r) {
  if (r <= YARN_ALPHA)
    return 0.0f;
  if (r >= YARN_BETA)
    return 1.0f;
  return (r - YARN_ALPHA) / (YARN_BETA - YARN_ALPHA);
}
static inline void apply_rope_yarn_heads(float *vec, int n_heads, int head_dim,
                                         int pos, float rope_base, float s,
                                         int L_new) {
  if (!(s > 0.0f))
    s = 1.0f;
  if (s > 1024.0f)
    s = 1024.0f;
  const int H = head_dim, HH = H >> 1;
  const float conc = (s > 1.0f) ? (1.0f + 0.1f * logf(s)) : 1.0f;
  const float Lorig = s * (float)L_new;

  for (int h = 0; h < n_heads; ++h) {
    float *b = vec + (size_t)h * H;
    for (int j = 0; j < HH; ++j) {
      float invf = powf(rope_base, -(2.0f * (float)j) / (float)H);
      float r = Lorig * invf;
      float g = yarn_gamma(r);
      float m = s * (1.0f - g) + 1.0f * g;
      float ang = (float)pos * (m * invf);
      float c = cosf(ang), sd = sinf(ang);
      float x0 = b[j], x1 = b[j + HH];
      b[j] = conc * (x0 * c - x1 * sd);
      b[j + HH] = conc * (x0 * sd + x1 * c);
    }
  }
}

// ---------- forward one token ----------
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float silu(float x) { return x * sigmoidf(x); }

static float *gptoss_forward(GPTOSSModel *m, int token, int pos) {
  GPTOSSConfig *p = &m->cfg;
  GPTOSSWeights *w = &m->w;
  RunState *s = &m->s;
  const int dim = p->dim, head_dim = s->head_dim, kv_dim = s->kv_dim,
            q_dim = s->q_dim, q_heads = s->q_heads;

  // token embedding
  memcpy(s->x, w->tok_embeddings + (size_t)token * dim, sizeof(float) * dim);

  for (int l = 0; l < p->n_layers; ++l) {
    // --- Attention ---
    rmsnorm(s->xb, s->x, w->rms_att_w + (size_t)l * dim, dim);

    matmul(s->q, s->xb, w->wq + (size_t)l * (size_t)q_dim * dim, dim, q_dim);
    matmul(s->k, s->xb, w->wk + (size_t)l * (size_t)kv_dim * dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + (size_t)l * (size_t)kv_dim * dim, dim, kv_dim);

    for (int i = 0; i < q_dim; i++)
      s->q[i] += w->bq[(size_t)l * q_dim + i];
    for (int i = 0; i < kv_dim; i++)
      s->k[i] += w->bk[(size_t)l * kv_dim + i];
    for (int i = 0; i < kv_dim; i++)
      s->v[i] += w->bv[(size_t)l * kv_dim + i];

    apply_rope_yarn_heads(s->q, q_heads, head_dim, pos, p->rope_base,
                          p->rope_scale, p->seq_len);
    apply_rope_yarn_heads(s->k, p->n_kv_heads, head_dim, pos, p->rope_base,
                          p->rope_scale, p->seq_len);

    // KV cache write
    const size_t loff =
        (size_t)l * p->seq_len * (size_t)kv_dim + (size_t)pos * (size_t)kv_dim;
    memcpy(s->key_cache + loff, s->k, kv_dim * sizeof(float));
    memcpy(s->val_cache + loff, s->v, kv_dim * sizeof(float));

    // banded window?
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

    memset(s->xq, 0, (size_t)q_dim * sizeof(float));

    for (int h = 0; h < q_heads; ++h) {
      float *qh = s->q + (size_t)h * head_dim;
      float *att = s->att + (size_t)h * p->seq_len;

      for (int t = t_start; t <= pos; ++t) {
        const float *krow =
            s->key_cache + (size_t)l * p->seq_len * (size_t)kv_dim +
            (size_t)t * (size_t)kv_dim + (size_t)(h % p->n_kv_heads) * head_dim;
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++)
          score += qh[i] * krow[i];
        att[t] = score / sqrtf((float)head_dim);
      }

      // softmax over [t_start..pos] with learned sink in denominator
      float mx = -1e30f;
      for (int t = t_start; t <= pos; ++t)
        if (att[t] > mx)
          mx = att[t];
      float sum = 0.0f;
      for (int t = t_start; t <= pos; ++t) {
        float e = expf(att[t] - mx);
        att[t] = e;
        sum += e;
      }
      float sink =
          w->attn_sink[(size_t)l * (size_t)p->n_heads + (h % p->n_heads)];
      sum += expf(sink - mx);
      float inv = 1.0f / sum;
      for (int t = t_start; t <= pos; ++t)
        att[t] *= inv;

      float *out = s->xq + (size_t)h * head_dim;
      for (int t = t_start; t <= pos; ++t) {
        const float a = att[t];
        const float *vrow =
            s->val_cache + (size_t)l * p->seq_len * (size_t)kv_dim +
            (size_t)t * (size_t)kv_dim + (size_t)(h % p->n_kv_heads) * head_dim;
        for (int i = 0; i < head_dim; i++)
          out[i] += a * vrow[i];
      }
    }

    // out proj + bias + residual
    matmul(s->xb2, s->xq, w->wo + (size_t)l * (size_t)dim * (size_t)q_dim,
           q_dim, dim);
    for (int i = 0; i < dim; i++)
      s->x[i] += s->xb2[i] + w->bo[(size_t)l * dim + i];

    // --- FFN (dense in your checkpoint) ---
    rmsnorm(s->xb, s->x, w->rms_ffn_w + (size_t)l * dim, dim);

    if (p->n_experts > 0) {
      // Router logits r[E] = WR[E,dim] @ xb + BR[E]
      const int E = p->n_experts, H = p->hidden_dim;
      float *r = s->router;
      for (int e = 0; e < E; e++) {
        const float *we = w->wr + (size_t)l * E * dim + (size_t)e * dim;
        float sum = 0.0f;
        for (int i = 0; i < dim; i++)
          sum += we[i] * s->xb[i];
        r[e] = sum + w->br[(size_t)l * E + e];
      }
      // pick top-k
      int tk = p->top_k > 0 && p->top_k < E ? p->top_k : (E < 4 ? E : 4);
      topk_indices(r, E, tk, s->topk_idx, s->topk_weight);
      // softmax over selected logits
      float mx = s->topk_weight[0];
      for (int i = 1; i < tk; i++)
        if (s->topk_weight[i] > mx)
          mx = s->topk_weight[i];
      float Z = 0.0f;
      for (int i = 0; i < tk; i++) {
        s->topk_weight[i] = expf(s->topk_weight[i] - mx);
        Z += s->topk_weight[i];
      }
      for (int i = 0; i < tk; i++)
        s->topk_weight[i] /= Z;

      // mixture accumulation
      memset(s->hb2, 0, sizeof(float) * dim); // reuse hb2(dim) as output acc
      for (int j = 0; j < tk; j++) {
        const int e = s->topk_idx[j];
        const float mix = s->topk_weight[j];

        // up = WUP[e] @ xb + bup[e]   [H]
        float *up = s->hb;
        memset(up, 0, sizeof(float) * H);
        const float *WUP =
            w->wup + ((size_t)l * E + e) * (size_t)H * (size_t)dim;
        for (int h = 0; h < H; ++h) {
          const float *row = WUP + (size_t)h * dim;
          float ssum = 0.0f;
          for (int i = 0; i < dim; i++)
            ssum += row[i] * s->xb[i];
          up[h] = ssum + w->bup[(size_t)l * E * H + (size_t)e * H + h];
        }

        // gate = WGATE[e] @ xb + bgate[e] ; act = silu(gate) * up   [H]
        const float *WG =
            w->wgate + ((size_t)l * E + e) * (size_t)H * (size_t)dim;
        for (int h = 0; h < H; ++h) {
          const float *row = WG + (size_t)h * dim;
          float gsum = 0.0f;
          for (int i = 0; i < dim; i++)
            gsum += row[i] * s->xb[i];
          float g = gsum + w->bgate[(size_t)l * E * H + (size_t)e * H + h];
          up[h] = silu(g) * up[h];
        }

        // down = WDOWN[e] @ act + bdown[e]   [dim]
        const float *WD =
            w->wdown + ((size_t)l * E + e) * (size_t)dim * (size_t)H;
        for (int i = 0; i < dim; i++) {
          const float *row = WD + (size_t)i * H;
          float dsum = 0.0f;
          for (int h = 0; h < H; ++h)
            dsum += row[h] * up[h];
          s->xb2[i] =
              dsum + w->bdown[(size_t)l * E * dim + (size_t)e * dim + i];
        }
        for (int i = 0; i < dim; i++)
          s->hb2[i] += mix * s->xb2[i];
      }

      // residual
      for (int i = 0; i < dim; i++)
        s->x[i] += s->hb2[i];

    } else {
      // dense path not present in this checkpoint; keep as no-op
    }
  }

  // final norm + logits
  rmsnorm(s->x, s->x, w->rms_final_w, dim);
  matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
  return s->logits;
}

// ---------- free ----------
static void gptoss_free(GPTOSSModel *m) {
  if (m->mapped && m->mapped != MAP_FAILED)
    munmap(m->mapped, m->file_size);
  if (m->fd != -1)
    close(m->fd);
  free_run_state(&m->s);
}

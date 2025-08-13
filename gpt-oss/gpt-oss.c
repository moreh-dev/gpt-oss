// gpt-oss.c — Minimal CPU-only GPT-OSS inference (single process)
// Build:   cc -O3 -march=native -o gpt-oss gpt-oss.c -lm
// Run:     ./gpt-oss gpt-oss-20b.bin -i "Hello" -n 64 -t tokenizer.bin
//
// Notes:
//  - Inputs rely on a custom .bin created by your exporter to match the layout
//  below.
//  - Tokenization uses your existing tokenizer.{c,h} (o200k_harmony).
//  - This is a didactic, correctness-first implementation (no threading/AVX).
//
// Features:
//  - Grouped-Query Attention (n_heads, n_kv_heads)
//  - Alternating dense vs. sliding-window attention (configurable via
//  alt_banded/window)
//  - Learned attention-sink term (adds an extra column to the softmax
//  denominator only)
//  - RoPE with YaRN-style scaling
//  - MoE FFN (top-k) with SwiGLU (+ clamp) and +1 bias on the linear (gate)
//  branch
//  - Greedy sampling (argmax)
//
// Header contract (binary):
//  - File starts with a GPTOSSConfig struct (fixed-size, little-endian).
//  - Followed by all weights as contiguous float32 in the order mapped in
//  memory_map_weights().
//  - If vocab_size in header is negative => output head tied to embeddings
//  (|vocab_size| is true size).

#include "tokenizer.h"

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
// Model config (must match exporter)

typedef struct {
  int dim;         // residual dimension (model width)
  int hidden_dim;  // MoE expert hidden dimension (per expert)
  int n_layers;    // transformer depth
  int n_heads;     // number of Q heads
  int n_kv_heads;  // number of KV heads (GQA)
  int n_experts;   // experts per layer (e.g., 32 for 20B, 128 for 120B)
  int top_k;       // router top-k (usually 4)
  int vocab_size;  // tokenizer vocab; negative => tied output head
  int seq_len;     // max sequence length in checkpoint
  int window;      // sliding window size (e.g., 128); 0 => fully dense
  int alt_banded;  // 0=dense only, +1=even banded, -1=odd banded, +2=all banded
  float rope_base; // RoPE theta (e.g., 150000.0)
  float rope_scale;     // YaRN scaling factor (>0)
  float rope_ntk_alpha; // optional (kept for completeness)
  float rope_ntk_beta;  // optional (kept for completeness)
  float swiglu_limit;   // clamp for pre-activations (e.g., 7.0)
} GPTOSSConfig;

// -------------------------------
// Weights layout (contiguous floats)

typedef struct {
  // token embedding
  float *tok_embeddings; // (vocab_size, dim)

  // per-layer rmsnorm weights
  float *rms_att_w; // (n_layers, dim)
  float *rms_ffn_w; // (n_layers, dim)

  // attention projections (with biases)
  float *wq; // (n_layers, dim, dim)
  float *wk; // (n_layers, dim, kv_dim)
  float *wv; // (n_layers, dim, kv_dim)
  float *wo; // (n_layers, dim, dim)  [stored row-major as (out, in)]
  float *bq; // (n_layers, dim)
  float *bk; // (n_layers, kv_dim)
  float *bv; // (n_layers, kv_dim)
  float *bo; // (n_layers, dim)

  // learned attention-sink bias (denominator-only), per head
  float *attn_sink; // (n_layers, n_heads)

  // MoE router affine (logits over experts)
  float *wr; // (n_layers, n_experts, dim)
  float *br; // (n_layers, n_experts)

  // MoE experts (SwiGLU)
  // W_up:   (hidden, dim)  ; b_up:   (hidden)
  // W_gate: (hidden, dim)  ; b_gate: (hidden)
  // W_down: (dim, hidden)  ; b_down: (dim)
  float *w_up;   // (n_layers, n_experts, hidden, dim)
  float *w_gate; // (n_layers, n_experts, hidden, dim)
  float *w_down; // (n_layers, n_experts, dim, hidden)
  float *b_up;   // (n_layers, n_experts, hidden)
  float *b_gate; // (n_layers, n_experts, hidden)
  float *b_down; // (n_layers, n_experts, dim)

  // final norm + LM head
  float *rms_final_w; // (dim)
  float *wcls;        // (vocab_size, dim) or tied to tok_embeddings
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

  float *att;    // (n_heads * seq_len) scratch for attention probabilities
  float *logits; // (vocab_size)

  // KV cache: (n_layers, seq_len, kv_dim)
  float *key_cache;
  float *val_cache;

  // router scratch
  float *router;      // (n_experts)
  int *topk_idx;      // (top_k)
  float *topk_weight; // (top_k) softmax over top-k
} RunState;

// -------------------------------
// Model holder

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
// Helpers

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

static inline void matmul(float *restrict y, const float *restrict x,
                          const float *restrict w, int n, int d) {
  // y(d) = W(d,n) @ x(n)
  for (int i = 0; i < d; i++) {
    float sum = 0.0f;
    const float *restrict wi = w + (size_t)i * n;
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
    const float v = x[i];
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

// Map weights from a single contiguous float* buffer. See exporter for packing
// order.
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
  w->wcls =
      shared_weights ? w->tok_embeddings : ptr; // if tied, points to embeddings
}

// Checkpoint reader (mmap)
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

  // Map as bytes then cast; we offset past the header when mapping pointers
  void *bytes = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (bytes == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  *data = (float *)bytes;

  int shared = (cfg->vocab_size < 0) ? 1 : 0;
  cfg->vocab_size = abs(cfg->vocab_size);

  if (!(cfg->rope_scale > 0.0f && isfinite(cfg->rope_scale)))
    cfg->rope_scale = 1.0f;
  if (!(cfg->rope_base > 0.0f && isfinite(cfg->rope_base)))
    cfg->rope_base = 10000.0f;
  if (!(cfg->swiglu_limit > 0.0f && isfinite(cfg->swiglu_limit)))
    cfg->swiglu_limit = 7.0f;

  // Advance pointer past header
  float *ptr = (float *)((uint8_t *)(*data) + sizeof(GPTOSSConfig));
  memory_map_weights(w, cfg, ptr, shared);
}

static void build_model(GPTOSSModel *m, const char *checkpoint) {
  m->fd = -1;
  m->data = NULL;
  m->file_size = 0;
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
// YaRN-style NTK-by-parts scaling

#define YARN_ALPHA 1.0f
#define YARN_BETA 32.0f

static inline float yarn_gamma(float r) {
  if (r <= YARN_ALPHA)
    return 0.0f;
  if (r >= YARN_BETA)
    return 1.0f;
  return (r - YARN_ALPHA) / (YARN_BETA - YARN_ALPHA);
}

// Apply RoPE (Q/K) with a simple YaRN-style schedule
static inline void apply_rope_yarn(float *vec, int size, int head_size, int pos,
                                   float rope_base, float s, int L_new) {
  if (!(s > 0.0f))
    s = 1.0f;
  if (s > 1024.0f)
    s = 1024.0f;

  const float concentration = (s > 1.0f) ? (1.0f + 0.1f * logf(s)) : 1.0f;

  // vec layout: [head0: head_size][head1: head_size]...; even/odd dims are
  // (cos,sin) pairs
  for (int i = 0; i < size; i += 2) {
    const int j = (i % head_size) >> 1;
    const float inv_freq =
        powf(rope_base, -(2.0f * (float)j) / (float)head_size);

    const float L_orig = s * (float)L_new;
    const float r = L_orig * inv_freq;
    const float g = yarn_gamma(r);
    const float m = s * (1.0f - g) + 1.0f * g; // blend

    const float angle = (float)pos * (m * inv_freq);
    const float c = cosf(angle), sd = sinf(angle);

    const float v0 = vec[i], v1 = vec[i + 1];
    vec[i] = concentration * (v0 * c - v1 * sd);
    vec[i + 1] = concentration * (v0 * sd + v1 * c);
  }
}

// -------------------------------
/* Forward pass for a single token at position `pos`.
 * Returns pointer to logits (vocab_size).
 */
static float *forward(GPTOSSModel *model, int token, int pos) {
  GPTOSSConfig *p = &model->config;
  GPTOSSWeights *w = &model->weights;
  RunState *s = &model->state;

  const int dim = p->dim;
  const int head_size = dim / p->n_heads;
  const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

  // token embedding lookup
  memcpy(s->x, w->tok_embeddings + (size_t)token * dim, sizeof(float) * dim);

  for (int l = 0; l < p->n_layers; l++) {
    // ---- Attention block ----
    rmsnorm(s->xb, s->x, w->rms_att_w + (size_t)l * dim, dim);

    // q, k, v projections (+ biases)
    matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim, dim, dim);
    matmul(s->k, s->xb, w->wk + (size_t)l * dim * kv_dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + (size_t)l * dim * kv_dim, dim, kv_dim);

    for (int i = 0; i < dim; i++)
      s->q[i] += w->bq[(size_t)l * dim + i];
    for (int i = 0; i < kv_dim; i++)
      s->k[i] += w->bk[(size_t)l * kv_dim + i];
    for (int i = 0; i < kv_dim; i++)
      s->v[i] += w->bv[(size_t)l * kv_dim + i];

    // RoPE (Q/K) + YaRN
    apply_rope_yarn(s->q, dim, head_size, pos, p->rope_base, p->rope_scale,
                    p->seq_len);
    apply_rope_yarn(s->k, kv_dim, head_size, pos, p->rope_base, p->rope_scale,
                    p->seq_len);

    // write into KV cache for current position
    const size_t loff = (size_t)l * p->seq_len * kv_dim + (size_t)pos * kv_dim;
    memcpy(s->key_cache + loff, s->k, kv_dim * sizeof(float));
    memcpy(s->val_cache + loff, s->v, kv_dim * sizeof(float));

    // Decide sliding-window vs dense for this layer
    int apply_window = 0;
    if (p->window > 0) {
      if (p->alt_banded == 0)
        apply_window = 0; // dense only
      else if (p->alt_banded == 2)
        apply_window = 1; // all banded
      else if (p->alt_banded > 0)
        apply_window = ((l % 2) == 0); // even banded
      else
        apply_window = ((l % 2) == 1); // odd banded
    }

    int t_start = 0;
    if (apply_window) {
      int wstart = pos - (p->window - 1);
      if (wstart > 0)
        t_start = wstart;
    }

    // clear attention output buffer
    memset(s->xb, 0, (size_t)dim * sizeof(float));

    for (int h = 0; h < p->n_heads; h++) {
      float *q = s->q + (size_t)h * head_size;
      float *att = s->att + (size_t)h * p->seq_len;

      // compute attention logits over active window
      for (int t = t_start; t <= pos; t++) {
        const float *krow = s->key_cache + (size_t)l * p->seq_len * kv_dim +
                            (size_t)t * kv_dim +
                            (size_t)(h % p->n_kv_heads) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
          score += q[i] * krow[i];
        att[t] = score / sqrtf((float)head_size);
      }

      // numerically-stable softmax over [t_start..pos] (+ denominator-only
      // sink)
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

      // learned attention sink acts like an extra column in the denominator
      const float sink = w->attn_sink[(size_t)l * p->n_heads + h];
      sum += expf(sink - mx);

      const float inv = 1.0f / sum;
      for (int t = t_start; t <= pos; t++)
        att[t] *= inv;

      // weighted sum of values
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

    // output projection + bias + residual
    matmul(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++)
      s->x[i] += s->xb2[i] + w->bo[(size_t)l * dim + i];

    // ---- MoE FFN block ----
    rmsnorm(s->xb, s->x, w->rms_ffn_w + (size_t)l * dim, dim);

    if (p->n_experts > 0 && p->top_k > 0) {
      // router logits -> softmax over top-k experts
      matmul(s->router, s->xb, w->wr + (size_t)l * p->n_experts * dim, dim,
             p->n_experts);
      for (int e = 0; e < p->n_experts; e++)
        s->router[e] += w->br[(size_t)l * p->n_experts + e];

      float topv[64]; // supports up to k<=64
      topk_indices(s->router, p->n_experts, p->top_k, s->topk_idx, topv);
      for (int i = 0; i < p->top_k; i++)
        s->topk_weight[i] = topv[i];
      softmax_inplace(s->topk_weight, p->top_k);

      // accumulate expert outputs: sum_r softmax_r * Down( SwiGLU(Up, Gate) )
      memset(s->xb2, 0, (size_t)dim * sizeof(float));

      const float alpha = 1.702f;          // SiLU multiplier (SwiGLU)
      const float limit = p->swiglu_limit; // clamp

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

        // Up and Gate projections (+ biases)
        matmul(s->hb, s->xb, w->w_up + up_off, p->dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++)
          s->hb[i] += w->b_up[bu_off + i];

        matmul(s->hb2, s->xb, w->w_gate + gate_off, p->dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++)
          s->hb2[i] += w->b_gate[bg_off + i];

        // Clamp pre-activations
        for (int i = 0; i < p->hidden_dim; i++) {
          float u = s->hb[i];
          float g = s->hb2[i];

          if (u > limit)
            u = limit; // up branch: clamp high only (matches ref)
          if (g > limit)
            g = limit; // gate branch: clamp both sides
          if (g < -limit)
            g = -limit;

          // SiLU(up) * (gate + 1)
          float silu = u * (1.0f / (1.0f + expf(-alpha * u)));
          s->hb[i] = silu * (g + 1.0f);
        }

        // Down projection (+ bias) into xb (temp)
        matmul(s->xb, s->hb, w->w_down + down_off, p->hidden_dim, p->dim);
        for (int i = 0; i < p->dim; i++)
          s->xb[i] += w->b_down[bd_off + i];

        // Weighted accumulation
        for (int i = 0; i < dim; i++)
          s->xb2[i] += w_r * s->xb[i];
      }

      // residual
      for (int i = 0; i < dim; i++)
        s->x[i] += s->xb2[i];
    } else {
      // (Optional) dense FFN fallback could be added here if a dense checkpoint
      // is used.
    }
  }

  // final rmsnorm + logits
  rmsnorm(s->x, s->x, w->rms_final_w, dim);
  matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
  return s->logits;
}

// -------------------------------
// Sampler (greedy)

typedef struct {
  int vocab_size;
} Sampler;

static inline int sample_argmax(const float *p, int n) {
  int mi = 0;
  float mv = p[0];
  for (int i = 1; i < n; i++)
    if (p[i] > mv) {
      mv = p[i];
      mi = i;
    }
  return mi;
}

static inline int sample_next(Sampler *s, float *logits) {
  (void)s;
  return sample_argmax(logits, s->vocab_size);
}

// -------------------------------
// CLI

#ifndef TESTING
static void usage(void) {
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

  // Tokenizer
  Tokenizer tokenizer;
  read_tokenizer(&tokenizer, tokpath, model.config.vocab_size);

  // Find special tokens (prefer end-of-message; fall back to end-of-text or
  // return)
  const char *TOK_START = "<|start|>";
  const char *TOK_END = "<|end|>";
  const char *TOK_EOT = "<|endoftext|>";
  const char *TOK_RETURN = "<|return|>";

  int BOS = find_token_id(&tokenizer, TOK_START, (int)strlen(TOK_START));
  int EOS = -1;
  int tmp = find_token_id(&tokenizer, TOK_END, (int)strlen(TOK_END));
  if (tmp >= 0)
    EOS = tmp;
  tmp = find_token_id(&tokenizer, TOK_EOT, (int)strlen(TOK_EOT));
  if (EOS < 0 && tmp >= 0)
    EOS = tmp;
  tmp = find_token_id(&tokenizer, TOK_RETURN, (int)strlen(TOK_RETURN));
  if (EOS < 0 && tmp >= 0)
    EOS = tmp;

  // Encode prompt (BOS if available; no EOS)
  int *tokens = (int *)malloc(sizeof(int) * (model.config.seq_len));
  int ntok = 0;
  encode(&tokenizer, prompt, BOS, -1, tokens, &ntok, model.config.seq_len);
  if (ntok < 1) {
    fprintf(stderr, "empty prompt after encoding\n");
    return 1;
  }

  Sampler sampler = {.vocab_size = model.config.vocab_size};

  // Run prompt to fill KV cache (print recovered text pieces)
  int pos = 0;
  int token = tokens[0];
  for (; pos < ntok - 1; pos++) {
    (void)forward(&model, token, pos);
    char *piece = decode_piece(&tokenizer, token, tokens[pos + 1]);
    safe_printf(piece);
    fflush(stdout);
    token = tokens[pos + 1];
  }

  // Generate continuation (new tokens only)
  int to_generate = steps;
  if (to_generate > model.config.seq_len - ntok)
    to_generate = model.config.seq_len - ntok;

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

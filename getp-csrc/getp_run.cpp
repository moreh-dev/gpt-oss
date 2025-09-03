// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <omp.h>

#include "tokenizer.hpp"

#ifndef GETP_RUN
#define GETP_RUN

// ------------------------------- Helpers ---------------------------------

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)e, hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

static inline long time_in_ms() {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------- Model structs -----------------------------

typedef struct {
  // Model Config
  int vocab_size;
  int hidden_dim;
  // MLP Config
  int n_experts;
  int experts_per_token;
  int intermediate_dim;
  int n_layers;
  // Attention Config
  int head_dim;
  int n_attn_heads;
  int n_kv_heads;
  int seq_len;
  int initial_context_length;
  float rope_theta;
  float rope_scaling_factor;
  int sliding_window;
  float swiglu_limit;
} Config;

typedef struct {
  // host pointers mapped from file
  float *token_embedding_table; // (vocab, hidden)
  float *rms_attn_w;            // (layers, hidden)
  float *rms_ffn_w;             // (layers, hidden)
  float *w_qkv;                 // (layers, (Hq+2Hk) * hidden)
  float *w_o;                   // (layers, hidden * (Hq))
  float *b_qkv;                 // (layers, Hq + 2Hk)
  float *b_o;                   // (layers, hidden)
  float *attn_sinks;            // (layers, n_attn_heads)
  float *w_router;              // (layers, hidden * n_experts)
  float *b_router;              // (layers, n_experts)
  float *w_mlp1;                // (layers, n_experts, 2*intermediate, hidden)
  float *w_mlp2;                // (layers, n_experts, hidden, intermediate)
  float *b_mlp1;                // (layers, n_experts, 2*intermediate)
  float *b_mlp2;                // (layers, n_experts, hidden)
  float *rms_out_w;             // (hidden)
  float *out;                   // (vocab, hidden)
} WeightsCPU;

typedef struct {
  // device buffers (mirrors of weights)
  float *token_embedding_table;
  float *rms_attn_w;
  float *rms_ffn_w;
  float *w_qkv;
  float *w_o;
  float *b_qkv;
  float *b_o;
  float *attn_sinks;
  float *w_router;
  float *b_router;
  float *w_mlp1;
  float *w_mlp2;
  float *b_mlp1;
  float *b_mlp2;
  float *rms_out_w;
  float *out;
} WeightsGPU;

typedef struct {
  // host pinned or device runtime buffers
  // On device:
  float *x;         // (hidden)
  float *t;         // (hidden)
  float *tb;        // (n_attn_heads * head_dim)
  float *tb2;       // (hidden)
  float *router;    // (n_experts)
  float *topk_v;    // (experts_per_token)
  int   *topk_i;    // (experts_per_token)
  float *mlp1_out;  // (2*intermediate)
  float *gate;      // (intermediate)
  float *up;        // (intermediate)
  float *gate_up;   // (intermediate)
  float *e_agg;     // (hidden)

  float *qkv;       // (head_dim * (n_attn_heads + 2*n_kv_heads))
  float *q;         // (n_attn_heads * head_dim)
  float *k_cur;     // view into kv cache for pos
  float *v_cur;     // same
  float *att;       // (n_attn_heads * (seq_len+1))  (+sink)
  float *logits;    // (vocab)

  // KV cache (device)
  float *key_cache;   // (layers * seq_len * kv_dim)
  float *value_cache; // (layers * seq_len * kv_dim)

  // mask for sliding window (device; optional)
  float *mask;        // (seq_len * seq_len) if sliding_window > 0
} RunStateGPU;

typedef struct {
  Config cfg;
  WeightsCPU wcpu;
  WeightsGPU wgpu;
  RunStateGPU s;
  int fd;
  float *mapped;     // host mapping
  ssize_t file_size;
} TransformerGPU;

// ------------------------------ HIP kernels ------------------------------

// (1) vector fill/zero
__global__ void k_set(float *x, float v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

// (2) elementwise scale (used in norms)
__global__ void k_scale(float *o, const float *x, const float *w, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = w[i] * (s * x[i]);
}

// (3) RMSNorm: compute inv_sqrt(mean(x^2)+eps) then scale by weight
__global__ void k_rms_reduce(const float *x, double *block_sums, int n) {
  extern __shared__ double ssum[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  double v = 0.0;
  if (idx < n) {
    float t = x[idx];
    v = (double)t * (double)t;
  }
  ssum[tid] = v;
  __syncthreads();
  // reduce
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) ssum[tid] += ssum[tid + s];
    __syncthreads();
  }
  if (tid == 0) block_sums[blockIdx.x] = ssum[0];
}

// (4) GEMV: y = W(in->out)[out,in] * x(in)
template<int TILE>
__global__ void k_gemv_rowmajor(const float * __restrict__ W,
                                const float * __restrict__ x,
                                float * __restrict__ y,
                                int in_features, int out_features) {
  // One block computes many out rows (by striding); each thread accumulates a partial
  int out = blockIdx.x * blockDim.x + threadIdx.x; // one thread per out row
  if (out >= out_features) return;

  const float *wrow = W + (size_t)out * in_features;
  float acc = 0.f;

  // tile over input dim
  for (int j0 = 0; j0 < in_features; j0 += TILE) {
    int j = j0 + threadIdx.y; // use y-threads to help coalesce x loads
    float xv = 0.f;
    if (j < in_features) xv = x[j];
    __syncthreads(); // not strictly needed with independent loads
    // each thread accumulates its own dot across the tile
    // but we simply stride by 1 here for simplicity and rely on L2
    for (int jj = j0; jj < min(j0+TILE, in_features); ++jj) {
      acc += wrow[jj] * x[jj];
    }
  }
  y[out] = acc;
}

// (5) add bias
__global__ void k_add_bias(float *y, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += b[i];
}

// (6) split qkv: q=(Hq*D), k=(Hk*D), v=(Hk*D)
__global__ void k_split_qkv(const float *qkv, float *q, float *k, float *v,
                            int head_dim, int n_q, int n_kv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = head_dim * (n_q + 2*n_kv);
  if (i >= total) return;
  if (i < head_dim * n_q) {
    q[i] = qkv[i];
  } else if (i < head_dim * (n_q + n_kv)) {
    k[i - head_dim * n_q] = qkv[i];
  } else {
    v[i - head_dim * (n_q + n_kv)] = qkv[i];
  }
}

// (7) RoPE for all heads in-place (q or k)
__global__ void k_rope(float *x, const float *cosv, const float *sinv,
                       int n_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim / 2;
  if (i >= half || h >= n_heads) return;
  float x1 = x[h*head_dim + i];
  float x2 = x[h*head_dim + half + i];
  float c = cosv[i];
  float s = sinv[i];
  float o1 = x1 * c - x2 * s;
  float o2 = x2 * c + x1 * s;
  x[h*head_dim + i] = o1;
  x[h*head_dim + half + i] = o2;
}

// (8) attention scores (dot(q, k[t])) + mask; per head
__global__ void k_attn_scores(const float *q, const float *k_cache,
                              const float *mask, float *att,
                              int head_dim, int kv_mul,
                              int seq_len, int pos, int kv_dim,
                              int n_heads, int sliding_window, int apply_sw_mask) {
  int h = blockIdx.x;     // head
  int t = threadIdx.x;    // timestep 0..pos (+1 for sink later)
  if (h >= n_heads || t > pos) return;

  const float *qh = q + h*head_dim;
  const float *kt = k_cache + t*kv_dim + (h/kv_mul)*head_dim;

  // dot
  float s = 0.f;
  #pragma unroll
  for (int i=0;i<head_dim;i++) s += qh[i]*kt[i];
  s /= sqrtf((float)head_dim);

  if (apply_sw_mask) {
    // mask[pos, t] already -inf where outside window
    s += mask[pos * seq_len + t];
  }
  att[h*(seq_len+1) + t] = s;
}

// (9) per-head softmax over length (pos+2) [0..pos and sink at pos+1]
__global__ void k_softmax_row(float *att, int row_len) {
  // one block per row, 1D thread over len
  extern __shared__ float sh[];
  float *buf = sh; // for reductions
  int tid = threadIdx.x;
  int n = row_len;

  // find max
  float mx = -INFINITY;
  for (int i = tid; i < n; i += blockDim.x) {
    mx = fmaxf(mx, att[i]);
  }
  buf[tid] = mx;
  __syncthreads();
  for (int s = blockDim.x/2; s>0; s>>=1) {
    if (tid < s) buf[tid] = fmaxf(buf[tid], buf[tid+s]);
    __syncthreads();
  }
  mx = buf[0];

  // exp & sum
  float sum = 0.f;
  for (int i = tid; i < n; i += blockDim.x) {
    float v = expf(att[i] - mx);
    att[i] = v;
    sum += v;
  }
  buf[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x/2; s>0; s>>=1) {
    if (tid < s) buf[tid] += buf[tid+s];
    __syncthreads();
  }
  sum = buf[0];

  // normalize
  for (int i = tid; i < n; i += blockDim.x) {
    att[i] = att[i] / sum;
  }
}

// (10) weighted sum over V cache → tb (per head)
__global__ void k_attn_weighted_sum(const float *att, const float *v_cache,
                                    float *tb, int head_dim, int kv_mul,
                                    int seq_len, int pos, int kv_dim,
                                    int n_heads) {
  int h = blockIdx.x;          // head
  int i = threadIdx.x;         // dim element
  if (h >= n_heads || i >= head_dim) return;

  float out = 0.f;
  for (int t=0; t<=pos; ++t) {
    const float a = att[h*(seq_len+1) + t];
    const float *vt = v_cache + t*kv_dim + (h/kv_mul)*head_dim;
    out += a * vt[i];
  }
  tb[h*head_dim + i] = out;
}

// (11) elementwise: gate_up (SwiGLU + clamp + extra bias 1.0 on up)
__global__ void k_swiglu_gate_up(float *gate, float *up, float *out,
                                 int n, float alpha, float clampv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  float u = up[i];
  // clamp
  g = fminf(g, clampv);
  u = fminf(fmaxf(u, -clampv), clampv);
  // silu
  g = g * (1.0f / (1.0f + expf(-alpha * g)));
  // up + 1
  u = u + 1.0f;
  out[i] = g * u;
}

// (12) vector add (residual)
__global__ void k_axpy(float *x, const float *y, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += alpha * y[i];
}

// ------------------------ GPU setup / allocations ------------------------

static void to_device(float **dptr, const float *hptr, size_t nbytes) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  HIP_CHECK(hipMemcpy(*dptr, hptr, nbytes, hipMemcpyHostToDevice));
}

static void alloc_device(float **dptr, size_t nbytes, float fill=0.f, bool set=false) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  if (set) {
    int n = (int)(nbytes / sizeof(float));
    int bs = 256, gs = (n + bs - 1) / bs;
    hipLaunchKernelGGL(k_set, dim3(gs), dim3(bs), 0, 0, *dptr, fill, n);
    HIP_CHECK(hipDeviceSynchronize());
  }
}

// ------------------------------- Math utils ------------------------------

static void rmsnorm_gpu(float *o, const float *x, const float *weight, int n) {
  const int BS = 256;
  int nb = (n + BS - 1)/BS;
  double *d_partials;
  HIP_CHECK(hipMalloc((void**)&d_partials, nb*sizeof(double)));
  hipLaunchKernelGGL(k_rms_reduce, dim3(nb), dim3(BS), BS*sizeof(double), 0, x, d_partials, n);
  HIP_CHECK(hipDeviceSynchronize());

  // reduce on host (nb is small)
  double hsum = 0.0;
  double *hparts = (double*)malloc(nb*sizeof(double));
  HIP_CHECK(hipMemcpy(hparts, d_partials, nb*sizeof(double), hipMemcpyDeviceToHost));
  for (int i=0;i<nb;i++) hsum += hparts[i];
  free(hparts);
  HIP_CHECK(hipFree(d_partials));

  double mean = hsum / (double)n;
  double inv = 1.0 / sqrt(mean + 1e-5);
  float s = (float)inv;

  int gs = (n + BS - 1)/BS;
  hipLaunchKernelGGL(k_scale, dim3(gs), dim3(BS), 0, 0, o, x, weight, s, n);
}

static void gemv_gpu(float *y, const float *x, const float *W, int in_features, int out_features) {
  // grid: 1D over out features; each thread handles a row
  const int TX = 256;
  dim3 block(TX, 1, 1);
  dim3 grid((out_features + TX - 1)/TX, 1, 1);
  hipLaunchKernelGGL((k_gemv_rowmajor<1024>), grid, block, 0, 0,
                     W, x, y, in_features, out_features);
}

static void add_bias_gpu(float *y, const float *b, int n) {
  const int BS=256, GS=(n+BS-1)/BS;
  hipLaunchKernelGGL(k_add_bias, dim3(GS), dim3(BS), 0, 0, y, b, n);
}

static void split_qkv_gpu(const float *qkv, float *q, float *k, float *v,
                          int head_dim, int n_q, int n_kv) {
  int total = head_dim*(n_q + 2*n_kv);
  const int BS=256, GS=(total+BS-1)/BS;
  hipLaunchKernelGGL(k_split_qkv, dim3(GS), dim3(BS), 0, 0, qkv, q, k, v,
                     head_dim, n_q, n_kv);
}

static void rope_gpu(float *x, const float *cosv, const float *sinv, int n_heads, int head_dim) {
  int half = head_dim/2;
  dim3 grid(n_heads);
  dim3 block( (unsigned)half );
  hipLaunchKernelGGL(k_rope, grid, block, 0, 0, x, cosv, sinv, n_heads, head_dim);
}

static void attn_scores_gpu(const float *q, const float *k_cache,
                            const float *mask, float *att,
                            int head_dim, int kv_mul, int seq_len, int pos,
                            int kv_dim, int n_heads, int sliding_window) {
  int apply = (sliding_window > 0) ? 1 : 0;
  // one block per head; threads over time steps (pad to warp multiple)
  int nthreads = 1; while (nthreads < (pos+1)) nthreads <<=1;
  if (nthreads < 64) nthreads = 64; // wavefront
  hipLaunchKernelGGL(k_attn_scores, dim3(n_heads), dim3(nthreads), 0, 0,
                     q, k_cache, mask, att,
                     head_dim, kv_mul, seq_len, pos, kv_dim, n_heads,
                     sliding_window, apply);
}

static void softmax_rows_gpu(float *att, int n_heads, int row_len) {
  // one block per row; use shared mem reductions
  for (int h=0; h<n_heads; ++h) {
    float *row = att + h*row_len;
    int bs = 256;
    hipLaunchKernelGGL(k_softmax_row, dim3(1), dim3(bs), bs*sizeof(float), 0, row, row_len);
  }
}

static void attn_weighted_sum_gpu(const float *att, const float *v_cache,
                                  float *tb, int head_dim, int kv_mul,
                                  int seq_len, int pos, int kv_dim, int n_heads) {
  dim3 grid(n_heads), block(head_dim);
  hipLaunchKernelGGL(k_attn_weighted_sum, grid, block, 0, 0,
                     att, v_cache, tb, head_dim, kv_mul, seq_len, pos, kv_dim, n_heads);
}

static void swiglu_gpu(float *gate, float *up, float *out,
                       int n, float alpha, float clampv) {
  const int BS=256, GS=(n+BS-1)/BS;
  hipLaunchKernelGGL(k_swiglu_gate_up, dim3(GS), dim3(BS), 0, 0, gate, up, out, n, alpha, clampv);
}

static void axpy_gpu(float *x, const float *y, float alpha, int n) {
  const int BS=256, GS=(n+BS-1)/BS;
  hipLaunchKernelGGL(k_axpy, dim3(GS), dim3(BS), 0, 0, x, y, alpha, n);
}

// ------------------------------- RoPE utils ------------------------------

static void compute_concentration_and_inv_freq_host(float base, int head_dim,
                                        float scaling_factor,
                                        float initial_context_length,
                                        float ntk_beta, float ntk_alpha,
                                        float *concentration_out,
                                        float *inv_freq_out) {
  int d_half = head_dim/2;
  float *freq = (float*)malloc(d_half*sizeof(float));
  #pragma omp parallel for
  for (int i=0;i<d_half;i++) freq[i] = powf(base, ((float)(2*i))/(float)head_dim);

  float concentration;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float low  = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
    float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
    assert(0 < low && low < high && high < d_half-1 + 1e-3);
    #pragma omp parallel for
    for (int i=0;i<d_half;i++) {
      float interpolation = 1.0f / (scaling_factor * freq[i]);
      float extrapolation = 1.0f / freq[i];
      float ramp = ((float)i - low) / (high - low);
      if (ramp < 0) ramp = 0;
      if (ramp > 1) ramp = 1;
      float mask = 1.0f - ramp;
      inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
    }
  } else {
    concentration = 1.0f;
    #pragma omp parallel for
    for (int i=0;i<d_half;i++) inv_freq_out[i] = 1.0f / freq[i];
  }
  *concentration_out = concentration;
  free(freq);
}

static void compute_cos_sin_host(int pos, float base, int head_dim,
                                 float scaling_factor, float initial_ctx,
                                 float ntk_beta, float ntk_alpha,
                                 float *cos_out, float *sin_out) {
  int d_half = head_dim/2;
  float conc;
  float *inv = (float*)malloc(d_half*sizeof(float));
  compute_concentration_and_inv_freq_host(base, head_dim, scaling_factor,
                                          initial_ctx, ntk_beta, ntk_alpha,
                                          &conc, inv);
  #pragma omp parallel for
  for (int j=0;j<d_half;j++) {
    float val = (float)pos * inv[j];
    cos_out[j] = cosf(val) * conc;
    sin_out[j] = sinf(val) * conc;
  }
  free(inv);
}

// ------------------------------- Sampler ---------------------------------

typedef struct {
  int vocab_size;
  float temperature;
  float topp;
  unsigned long long rng_state;
  struct ProbIndex { float prob; int index; } *probindex;
} Sampler;

static unsigned int random_u32(unsigned long long *s) {
  *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
  return (*s * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long *s) {
  return (random_u32(s) >> 8) / 16777216.0f;
}

static void build_sampler(Sampler *S, int vocab, float temp, float topp, unsigned long long seed) {
  S->vocab_size=vocab; S->temperature=temp; S->topp=topp; S->rng_state=seed;
  S->probindex = (Sampler::ProbIndex*)malloc(sizeof(Sampler::ProbIndex)*vocab);
}
static void free_sampler(Sampler *S){ free(S->probindex); }

static void softmax_host(float *x, int n) {
  float mx=x[0]; for(int i=1;i<n;i++) if (x[i]>mx) mx=x[i];
  double sum=0.0; for(int i=0;i<n;i++){ x[i] = expf(x[i]-mx); sum += x[i]; }
  for(int i=0;i<n;i++) x[i] /= (float)sum;
}
static int sample_argmax(const float *p, int n) {
  int m=0; float mv=p[0];
  for(int i=1;i<n;i++) if (p[i]>mv){mv=p[i]; m=i;}
  return m;
}
static int sample_mult(float *p, int n, float coin) {
  float c=0.f; for(int i=0;i<n;i++){ c+=p[i]; if (coin < c) return i; }
  return n-1;
}
static int cmp_probdesc(const void* A, const void* B) {
  const Sampler::ProbIndex *a=(const Sampler::ProbIndex*)A;
  const Sampler::ProbIndex *b=(const Sampler::ProbIndex*)B;
  return (a->prob > b->prob) ? -1 : (a->prob < b->prob);
}
static int sample_topp(float *p, int n, float topp, Sampler::ProbIndex *buf, float coin) {
  // crop tiny probs:
  int n0=0; const float cutoff=(1.0f - topp)/(n-1);
  for(int i=0;i<n;i++) if (p[i] >= cutoff){ buf[n0].index=i; buf[n0].prob=p[i]; n0++; }
  qsort(buf, n0, sizeof(buf[0]), cmp_probdesc);
  float cp=0.f; int last=n0-1;
  for(int i=0;i<n0;i++){ cp+=buf[i].prob; if (cp>topp){ last=i; break; } }
  float r = coin * cp, c=0.f;
  for(int i=0;i<=last;i++){ c+=buf[i].prob; if (r<c) return buf[i].index; }
  return buf[last].index;
}
static int sample_token(Sampler *S, float *logits) {
  int next;
  if (S->temperature==0.0f) {
    // argmax directly on logits
    return sample_argmax(logits, S->vocab_size);
  }
  for (int i=0;i<S->vocab_size;i++) logits[i] /= S->temperature;
  softmax_host(logits, S->vocab_size);
  float coin = random_f32(&S->rng_state);
  if (S->topp<=0 || S->topp>=1) return sample_mult(logits, S->vocab_size, coin);
  return sample_topp(logits, S->vocab_size, S->topp, S->probindex, coin);
}

// TopK on host (router)
static void topk_host(float *topk_v, int *topk_i, const float *scores, int n, int k) {
  // simple partial selection (n_experts is small-ish)
  for (int i=0;i<k;i++){ topk_v[i]=-INFINITY; topk_i[i]=-1; }
  for (int e=0;e<n;e++) {
    float v = scores[e];
    int slot = -1;
    for (int i=0;i<k;i++) if (v > topk_v[i]){ slot=i; break; }
    if (slot>=0) {
      for (int j=k-1;j>slot;j--){ topk_v[j]=topk_v[j-1]; topk_i[j]=topk_i[j-1]; }
      topk_v[slot]=v; topk_i[slot]=e;
    }
  }
  // softmax normalize top-k (in-place)
  softmax_host(topk_v, k);
}

// -------------------------- File mapping (host) --------------------------

static void memory_map_weights(WeightsCPU *w, const Config *c, float *ptr) {
  int head_dim = c->head_dim;
  int n_layers = c->n_layers;
  int n_experts = c->n_experts;

  w->token_embedding_table = ptr;
  ptr += 1ll * c->vocab_size * c->hidden_dim;

  w->out = ptr;
  ptr += 1ll * c->vocab_size * c->hidden_dim;

  w->rms_attn_w = ptr;
  ptr += 1ll * n_layers * c->hidden_dim;

  w->rms_ffn_w = ptr;
  ptr += 1ll * n_layers * c->hidden_dim;

  w->rms_out_w = ptr;
  ptr += 1ll * c->hidden_dim;

  w->w_qkv = ptr;
  ptr += 1ll * n_layers * c->hidden_dim *
         (head_dim * c->n_attn_heads + 2 * head_dim * c->n_kv_heads);

  w->b_qkv = ptr;
  ptr += 1ll * n_layers *
         (head_dim * c->n_attn_heads + 2 * head_dim * c->n_kv_heads);

  w->w_o = ptr;
  ptr += 1ll * n_layers * (head_dim * c->n_attn_heads) * c->hidden_dim;

  w->b_o = ptr;
  ptr += 1ll * n_layers * c->hidden_dim;

  w->attn_sinks = ptr;
  ptr += 1ll * n_layers * c->n_attn_heads;

  w->w_router = ptr;
  ptr += 1ll * n_layers * c->hidden_dim * n_experts;

  w->b_router = ptr;
  ptr += 1ll * n_layers * n_experts;

  w->w_mlp1 = ptr;
  ptr += 1ll * n_layers * n_experts * 2 * c->intermediate_dim * c->hidden_dim;

  w->b_mlp1 = ptr;
  ptr += 1ll * n_layers * n_experts * 2 * c->intermediate_dim;

  w->w_mlp2 = ptr;
  ptr += 1ll * n_layers * n_experts * c->hidden_dim * c->intermediate_dim;

  w->b_mlp2 = ptr;
  ptr += 1ll * n_layers * n_experts * c->hidden_dim;
}

static void load_checkpoint(const char *path, Config *cfg,
                            WeightsCPU *w, int *fd,
                            float **mapped, ssize_t *fsize) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "open %s failed\n", path); exit(1); }
  if (fread(cfg, sizeof(Config), 1, f) != 1) { fprintf(stderr, "bad header\n"); exit(1); }
  fseek(f, 0, SEEK_END);
  *fsize = ftell(f);
  fclose(f);

  *fd = open(path, O_RDONLY);
  if (*fd == -1) { fprintf(stderr, "open() failed\n"); exit(1); }

  *mapped = (float*)mmap(NULL, *fsize, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*mapped == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }

  float *weights_ptr = *mapped + (sizeof(Config) / sizeof(float));
  memory_map_weights(w, cfg, weights_ptr);

  // Log
  printf("[ckpt] vocab=%d hidden=%d layers=%d heads(q)=%d kv_heads=%d head_dim=%d seq_len=%d\n",
         cfg->vocab_size, cfg->hidden_dim, cfg->n_layers,
         cfg->n_attn_heads, cfg->n_kv_heads, cfg->head_dim, cfg->seq_len);
}

static void upload_weights(TransformerGPU *T) {
  const Config &c = T->cfg;
  const WeightsCPU &w = T->wcpu;
  WeightsGPU &g = T->wgpu;

  to_device(&g.token_embedding_table, w.token_embedding_table, 1ll*c.vocab_size*c.hidden_dim*sizeof(float));
  to_device(&g.out,                w.out,                1ll*c.vocab_size*c.hidden_dim*sizeof(float));
  to_device(&g.rms_attn_w,         w.rms_attn_w,         1ll*c.n_layers*c.hidden_dim*sizeof(float));
  to_device(&g.rms_ffn_w,          w.rms_ffn_w,          1ll*c.n_layers*c.hidden_dim*sizeof(float));
  to_device(&g.rms_out_w,          w.rms_out_w,          1ll*c.hidden_dim*sizeof(float));
  to_device(&g.w_qkv,              w.w_qkv,              1ll*c.n_layers*c.hidden_dim*(c.head_dim*c.n_attn_heads + 2*c.head_dim*c.n_kv_heads)*sizeof(float));
  to_device(&g.b_qkv,              w.b_qkv,              1ll*c.n_layers*(c.head_dim*c.n_attn_heads + 2*c.head_dim*c.n_kv_heads)*sizeof(float));
  to_device(&g.w_o,                w.w_o,                1ll*c.n_layers*(c.head_dim*c.n_attn_heads)*c.hidden_dim*sizeof(float));
  to_device(&g.b_o,                w.b_o,                1ll*c.n_layers*c.hidden_dim*sizeof(float));
  to_device(&g.attn_sinks,         w.attn_sinks,         1ll*c.n_layers*c.n_attn_heads*sizeof(float));
  to_device(&g.w_router,           w.w_router,           1ll*c.n_layers*c.hidden_dim*c.n_experts*sizeof(float));
  to_device(&g.b_router,           w.b_router,           1ll*c.n_layers*c.n_experts*sizeof(float));
  to_device(&g.w_mlp1,             w.w_mlp1,             1ll*c.n_layers*c.n_experts*2*c.intermediate_dim*c.hidden_dim*sizeof(float));
  to_device(&g.b_mlp1,             w.b_mlp1,             1ll*c.n_layers*c.n_experts*2*c.intermediate_dim*sizeof(float));
  to_device(&g.w_mlp2,             w.w_mlp2,             1ll*c.n_layers*c.n_experts*c.hidden_dim*c.intermediate_dim*sizeof(float));
  to_device(&g.b_mlp2,             w.b_mlp2,             1ll*c.n_layers*c.n_experts*c.hidden_dim*sizeof(float));
}

static void malloc_state(TransformerGPU *T) {
  const Config &c = T->cfg;
  RunStateGPU &s = T->s;

  alloc_device(&s.x,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.t,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.tb,       c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  alloc_device(&s.tb2,      c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.router,   c.n_experts*sizeof(float), 0.f, true);
  HIP_CHECK(hipMalloc((void**)&s.topk_v, c.experts_per_token*sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&s.topk_i, c.experts_per_token*sizeof(int)));
  alloc_device(&s.mlp1_out, 2*c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate,     c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.up,       c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate_up,  c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.e_agg,    c.hidden_dim*sizeof(float), 0.f, true);

  int qkv_tot = c.head_dim*(c.n_attn_heads + 2*c.n_kv_heads);
  alloc_device(&s.qkv,    qkv_tot*sizeof(float), 0.f, true);
  alloc_device(&s.q,      c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  // k_cur/v_cur views are offsets into caches (no alloc here)
  alloc_device(&s.att,    (c.n_attn_heads*(c.seq_len+1))*sizeof(float), 0.f, true);
  alloc_device(&s.logits, c.vocab_size*sizeof(float), 0.f, true);

  int kv_dim = c.head_dim * c.n_kv_heads;
  size_t cache_elems = 1ll*c.n_layers*c.seq_len*kv_dim;
  alloc_device(&s.key_cache,   cache_elems*sizeof(float), 0.f, true);
  alloc_device(&s.value_cache, cache_elems*sizeof(float), 0.f, true);

  if (c.sliding_window > 0) {
    alloc_device(&s.mask, 1ll*c.seq_len*c.seq_len*sizeof(float), 0.f, true);
    // host-init mask once then copy
    float *hmask = (float*)malloc(1ll*c.seq_len*c.seq_len*sizeof(float));
    for (int i=0;i<c.seq_len;i++) for (int j=0;j<c.seq_len;j++) {
      float v = 0.f;
      if (c.sliding_window > 0 && i - j >= c.sliding_window) v = -INFINITY;
      hmask[i*c.seq_len + j] = v;
    }
    HIP_CHECK(hipMemcpy(s.mask, hmask, 1ll*c.seq_len*c.seq_len*sizeof(float), hipMemcpyHostToDevice));
    free(hmask);
  } else {
    s.mask = nullptr;
  }
}

static void build_transformer(TransformerGPU *T, const char *ckpt) {
  T->fd = -1; T->mapped = nullptr; T->file_size = 0;
  hipSetDevice(0); // MI250 GCD0
  load_checkpoint(ckpt, &T->cfg, &T->wcpu, &T->fd, &T->mapped, &T->file_size);
  upload_weights(T);
  malloc_state(T);
}

// ------------------------------ I/O helpers ------------------------------

static void free_transformer(TransformerGPU *T) {
  if (T->mapped && T->mapped!=MAP_FAILED) munmap(T->mapped, T->file_size);
  if (T->fd!=-1) close(T->fd);

  // free device weights/state
  WeightsGPU &g = T->wgpu;
  auto F=[&](float *&p){ if(p){ hipFree(p); p=nullptr; } };
  F(g.token_embedding_table); F(g.rms_attn_w); F(g.rms_ffn_w); F(g.w_qkv); F(g.w_o);
  F(g.b_qkv); F(g.b_o); F(g.attn_sinks); F(g.w_router); F(g.b_router);
  F(g.w_mlp1); F(g.w_mlp2); F(g.b_mlp1); F(g.b_mlp2); F(g.rms_out_w); F(g.out);

  RunStateGPU &s = T->s;
  F(s.x); F(s.t); F(s.tb); F(s.tb2); F(s.router); if(s.topk_v) hipFree(s.topk_v);
  if(s.topk_i) hipFree(s.topk_i); F(s.mlp1_out); F(s.gate); F(s.up); F(s.gate_up); F(s.e_agg);
  F(s.qkv); F(s.q); F(s.att); F(s.logits); F(s.key_cache); F(s.value_cache);
  if (s.mask) hipFree(s.mask);
}

static void build_transformer(TransformerGPU *T, const char *ckpt) {
  T->fd = -1; T->mapped = nullptr; T->file_size = 0;
  hipSetDevice(0); // MI250 GCD0
  load_checkpoint(ckpt, &T->cfg, &T->wcpu, &T->fd, &T->mapped, &T->file_size);
  upload_weights(T);
  malloc_state(T);
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  char *checkpoint_path = "model.bin"; // e.g. out/model.bin
  const char *tokenizer_path = "tokenizer.bin";

  build_transformer(&transformer, checkpoint_path);
  read_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  free_transformer(&transformer);
  free_tokenizer(&tokenizer);
}

static float* forward(TransformerGPU *T, int token, int pos) {
  const Config &p = T->cfg;
  const WeightsGPU &w = T->wgpu;
  RunStateGPU &s = T->s;

  const int H = p.hidden_dim;
  const int D = p.head_dim;
  const int Hq = p.n_attn_heads;
  const int Hkv = p.n_kv_heads;
  const int kv_dim = D * Hkv;
  const int kv_mul = Hq / Hkv;

  // x = embedding[token]
  // We'll just gemv against an implicit one-hot by copying that row:
  // Copy happens on device: y = token_embedding_table[token, :]
  const float *emb_row = w.token_embedding_table + (size_t)token*H;
  HIP_CHECK(hipMemcpy(s.x, emb_row, H*sizeof(float), hipMemcpyDeviceToDevice));

  for (int l=0; l<p.n_layers; ++l) {
    // --- Attention RMSNorm: t = rmsnorm(x, rms_attn_w[l])
    rmsnorm_gpu(s.t, s.x, w.rms_attn_w + (size_t)l*H, H);

    // --- QKV projection: qkv = W_qkv[l] * t + b_qkv[l]
    const float *Wqkv = w.w_qkv + (size_t)l*H*(D*Hq + 2*D*Hkv);
    const float *Bqkv = w.b_qkv + (size_t)l*(D*Hq + 2*D*Hkv);
    gemv_gpu(s.qkv, s.t, Wqkv, H, (D*Hq + 2*D*Hkv));
    add_bias_gpu(s.qkv, Bqkv, (D*Hq + 2*D*Hkv));

    // split to q,k,v. k/v current position also appended into cache
    float *k_buf = T->s.key_cache + (size_t)l*p.seq_len*kv_dim + (size_t)pos*kv_dim;
    float *v_buf = T->s.value_cache + (size_t)l*p.seq_len*kv_dim + (size_t)pos*kv_dim;
    split_qkv_gpu(s.qkv, s.q, k_buf, v_buf, D, Hq, Hkv);

    // --- RoPE for q and k(pos)
    // compute cos/sin on host (cheap) then upload temporary
    int half = D/2;
    float *hcos = (float*)malloc(half*sizeof(float));
    float *hsin = (float*)malloc(half*sizeof(float));
    compute_cos_sin_host(pos, p.rope_theta, D, p.rope_scaling_factor,
                         p.initial_context_length, 32.0f, 1.0f, hcos, hsin);
    float *dcos, *dsin;
    HIP_CHECK(hipMalloc((void**)&dcos, half*sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dsin, half*sizeof(float)));
    HIP_CHECK(hipMemcpy(dcos, hcos, half*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsin, hsin, half*sizeof(float), hipMemcpyHostToDevice));
    free(hcos); free(hsin);

    rope_gpu(s.q, dcos, dsin, Hq, D);
    rope_gpu(k_buf, dcos, dsin, Hkv, D);
    HIP_CHECK(hipFree(dcos)); HIP_CHECK(hipFree(dsin));

    // --- Attention scores for all heads vs 0..pos
    float *k_layer_cache = T->s.key_cache + (size_t)l*p.seq_len*kv_dim;
    float *v_layer_cache = T->s.value_cache + (size_t)l*p.seq_len*kv_dim;
    attn_scores_gpu(s.q, k_layer_cache, (p.sliding_window>0 && (l % 2 == 0))? s.mask: nullptr, s.att,
                    D, kv_mul, p.seq_len, pos, kv_dim, Hq, (l%2==0)?p.sliding_window:0);

    // write sink score at index pos+1 (per head) from attn_sinks
    // We'll do it from host by staging a small vector:
    // step 1: copy att row for each head (pos+2 length) after kernel
    // simpler: small kernelless copy from host:
    float *h_sinks = (float*)malloc(Hq*sizeof(float));
    HIP_CHECK(hipMemcpy(h_sinks, w.attn_sinks + (size_t)l*Hq, Hq*sizeof(float), hipMemcpyDeviceToHost));
    // we just append sink value at [pos+1]
    for (int h=0; h<Hq; ++h) {
      float sinkv = h_sinks[h];
      HIP_CHECK(hipMemcpy(s.att + h*(p.seq_len+1) + (pos+1),
                          &sinkv, sizeof(float), hipMemcpyHostToDevice));
    }
    free(h_sinks);

    // softmax over len = pos+2 for each head
    softmax_rows_gpu(s.att, Hq, pos+2);

    // weighted sum over V → tb(heads*D)
    attn_weighted_sum_gpu(s.att, v_layer_cache, s.tb, D, kv_mul, p.seq_len, pos, kv_dim, Hq);

    // output projection: tb2 = W_o[l] * tb + b_o[l]
    const float *Wo = w.w_o + (size_t)l*(D*Hq)*H;
    const float *Bo = w.b_o + (size_t)l*H;
    gemv_gpu(s.tb2, s.tb, Wo, D*Hq, H);
    add_bias_gpu(s.tb2, Bo, H);

    // residual: x += tb2
    axpy_gpu(s.x, s.tb2, 1.0f, H);

    // --- MLP: t = rmsnorm(x, rms_ffn_w[l])
    rmsnorm_gpu(s.t, s.x, w.rms_ffn_w + (size_t)l*H, H);

    // router: router = W_router * t + b_router
    const float *Wr = w.w_router + (size_t)l*H*p.n_experts;
    const float *Br = w.b_router + (size_t)l*p.n_experts;
    gemv_gpu(s.router, s.t, Wr, H, p.n_experts);
    add_bias_gpu(s.router, Br, p.n_experts);
    HIP_CHECK(hipDeviceSynchronize()); // ensure router on device finished

    // Bring router to host → top-k selection on CPU (simple, correct)
    float *h_router = (float*)malloc(p.n_experts*sizeof(float));
    float *h_topk_v = (float*)malloc(p.experts_per_token*sizeof(float));
    int   *h_topk_i = (int*)  malloc(p.experts_per_token*sizeof(int));
    HIP_CHECK(hipMemcpy(h_router, s.router, p.n_experts*sizeof(float), hipMemcpyDeviceToHost));
    topk_host(h_topk_v, h_topk_i, h_router, p.n_experts, p.experts_per_token);
    HIP_CHECK(hipMemcpy(s.topk_v, h_topk_v, p.experts_per_token*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s.topk_i, h_topk_i, p.experts_per_token*sizeof(int),   hipMemcpyHostToDevice));
    free(h_router); free(h_topk_v); free(h_topk_i);

    // e_agg = 0
    {
      const int BS=256, GS=(H+BS-1)/BS;
      hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, s.e_agg, 0.f, H);
    }

    // For each selected expert e:
    for (int idx=0; idx<p.experts_per_token; ++idx) {
      int e;
      HIP_CHECK(hipMemcpy(&e, s.topk_i+idx, sizeof(int), hipMemcpyDeviceToHost));
      float wexp;
      HIP_CHECK(hipMemcpy(&wexp, s.topk_v+idx, sizeof(float), hipMemcpyDeviceToHost));

      // MLP1: mlp1_out = W_mlp1[l,e] * t + b_mlp1[l,e] → size 2*intermediate
      const float *W1 = w.w_mlp1 + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim) * H;
      const float *B1 = w.b_mlp1 + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim);
      gemv_gpu(s.mlp1_out, s.t, W1, H, 2*p.intermediate_dim);
      add_bias_gpu(s.mlp1_out, B1, 2*p.intermediate_dim);

      // split into gate/up (strided memcopy on device is okay via kernels, but here do 2 gemv-free copies)
      HIP_CHECK(hipMemcpy2D(s.gate, sizeof(float),
                            s.mlp1_out, sizeof(float)*2,
                            sizeof(float), p.intermediate_dim, hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy2D(s.up, sizeof(float),
                            s.mlp1_out+1, sizeof(float)*2,
                            sizeof(float), p.intermediate_dim, hipMemcpyDeviceToDevice));

      // SwiGLU + clamp → gate_up
      swiglu_gpu(s.gate, s.up, s.gate_up, p.intermediate_dim, 1.702f, p.swiglu_limit);

      // MLP2: tb2 = W_mlp2[l,e] * gate_up + b_mlp2[l,e] → size hidden
      const float *W2 = w.w_mlp2 + (size_t)(l*p.n_experts + e) * H * p.intermediate_dim;
      const float *B2 = w.b_mlp2 + (size_t)(l*p.n_experts + e) * H;
      gemv_gpu(s.tb2, s.gate_up, W2, p.intermediate_dim, H);
      add_bias_gpu(s.tb2, B2, H);

      // e_agg += tb2 * wexp
      axpy_gpu(s.e_agg, s.tb2, wexp, H);
    }

    // residual: x += e_agg
    axpy_gpu(s.x, s.e_agg, 1.0f, H);
  }

  // final rmsnorm
  rmsnorm_gpu(s.x, s.x, w.rms_out_w, H);

  // logits = out * x
  gemv_gpu(s.logits, s.x, w.out, H, p.vocab_size);
  HIP_CHECK(hipDeviceSynchronize());
  return s.logits;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  // <|start|>: 200006
  // <|end|>: 200007
  // <|return|>: 200002
  // <|message|>: 200008
  // <|channel|>: 200005
  // <|constrain|>: 200003
  // <|endoftext|>: 199999

  // Inference here

  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  // print the very first token
  // should be removed
  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece);
  fflush(stdout);

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    pos++;
    if (pos < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens] = next;
    }

    // data-dependent terminating condition: the EOS (=199999 or =200002) token
    // delimits sequences
    if (next == 199999 || next == 200002) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);

    token = next;
  }

  // should be removed
  // printf("\n");

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN

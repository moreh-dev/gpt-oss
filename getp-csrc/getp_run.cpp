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

#include "../tokenizer.hpp"

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

static void softmax_rows_gpu(float *att, int n_heads, int row_len, int row_stride) {
  const int BS = 256;
  for (int h=0; h<n_heads; ++h) {
    float *row = att + (size_t)h * row_stride; 
    hipLaunchKernelGGL(k_softmax_row, dim3(1), dim3(BS), BS*sizeof(float), 0, row, row_len);
  }
}

static void attn_weighted_sum_gpu(const float *att, const float *v_cache,
                                  float *tb, int head_dim, int kv_mul,
                                  int seq_len, int pos, int kv_dim, int n_heads) {
  int row_len = pos + 2;           
  int row_stride = seq_len + 1;
  dim3 grid(n_heads), block(head_dim);
  hipLaunchKernelGGL(k_attn_weighted_sum, grid, block, 0, 0,
                     att, v_cache, tb, head_dim, kv_mul,
                     row_stride, pos, kv_dim, n_heads);
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

static int cmp_probdesc(const void* A, const void* B) {
  const ProbIndex *a=(const ProbIndex*)A;
  const ProbIndex *b=(const ProbIndex*)B;
  return (a->prob > b->prob) ? -1 : (a->prob < b->prob);
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

void memory_map_weights_gpu(TransformerWeights *w, Config *cfg, float *ptr) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;

  to_device(&w->token_embedding_table, ptr, 1ll*cfg->vocab_size*cfg->hidden_dim*sizeof(float));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  to_device(&w->out, ptr, 1ll*cfg->vocab_size*cfg->hidden_dim*sizeof(float));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  to_device(&w->rms_attn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_ffn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_out_w, ptr, 1ll * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * cfg->hidden_dim;
  // hey it's qkvqkv, not qqkkvv
  to_device(&w->w_qkv, ptr,
            1ll * n_layers * cfg->hidden_dim *
            (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  to_device(&w->b_qkv, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  to_device(&w->w_o, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim *
            sizeof(float));
  ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
  to_device(&w->b_o, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->attn_sinks, ptr, 1ll * n_layers * cfg->n_attn_heads * sizeof(float));
  ptr += 1ll * n_layers * cfg->n_attn_heads;
  to_device(&w->w_router, ptr, 1ll * n_layers * cfg->hidden_dim * n_experts * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
  to_device(&w->b_router, ptr, 1ll * n_layers * n_experts * sizeof(float));
  ptr += 1ll * n_layers * n_experts;
  // hey it's gate_upgate_up, not gategateupup
  to_device(&w->w_mlp1, ptr,
            1ll * n_layers * n_experts * cfg->hidden_dim * 2 * cfg->intermediate_dim *
            sizeof(float));
  ptr +=
      1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;
  to_device(&w->b_mlp1, ptr, 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * sizeof(float));
  ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;
  to_device(&w->w_mlp2, ptr,
            1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim *
            sizeof(float));
  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;
  to_device(&w->b_mlp2, ptr, 1ll * n_layers * n_experts * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
}

void load_checkpoint_gpu(char *ckpt, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size) {
  FILE *file = fopen(ckpt, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", ckpt);
    exit(EXIT_FAILURE);
  }

  // read in the config header
  // load sizeof(Config) bytes into config
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // figure out the file size
  printf("vocab_size: %d\n", config->vocab_size);
  printf("hidden_dim: %d\n", config->hidden_dim);
  printf("n_experts: %d\n", config->n_experts);
  printf("experts_per_token: %d\n", config->experts_per_token);
  printf("intermediate_dim: %d\n", config->intermediate_dim);
  printf("n_layers: %d\n", config->n_layers);
  printf("head_dim: %d\n", config->head_dim);
  printf("n_attn_heads: %d\n", config->n_attn_heads);
  printf("n_kv_heads: %d\n", config->n_kv_heads);
  printf("max_seq_len: %d\n", config->seq_len);
  printf("init context len: %d\n", config->initial_context_length);
  printf("rope theta: %f\n", config->rope_theta);
  printf("rope_scaling_factor: %f\n", config->rope_scaling_factor);
  printf("sliding window: %d\n", config->sliding_window);
  printf("swiglu_limit: %f\n", config->swiglu_limit);
  fseek(file, 0, SEEK_END); // move file pointer to end of file

  *file_size = ftell(file); // get the file size, in bytes
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(ckpt, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed\n");
    exit(EXIT_FAILURE);
  }
  *data = reinterpret_cast<float *>(
      mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0));
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  float *weights_ptr = *data + sizeof(Config) / sizeof(float);
  memory_map_weights_gpu(weights, config, weights_ptr);
}

static void malloc_state_gpu(Transformer *T) {
  const Config &c = T->config;
  RunState &s = T->state;

  alloc_device(&s.x,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.t,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.tb,       c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  alloc_device(&s.tb2,      c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.router_score,   c.n_experts*sizeof(float), 0.f, true);
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

// ------------------------------ I/O helpers ------------------------------

static void free_transformer_gpu(Transformer *T) {
  if (T->data && T->data!=MAP_FAILED) munmap(T->data, T->file_size);
  if (T->fd!=-1) close(T->fd);

  // free device weights/state
  TransformerWeights &g = T->weights;
  auto F=[&](float *&p){ if(p){ hipFree(p); p=nullptr; } };
  F(g.token_embedding_table); F(g.rms_attn_w); F(g.rms_ffn_w); F(g.w_qkv); F(g.w_o);
  F(g.b_qkv); F(g.b_o); F(g.attn_sinks); F(g.w_router); F(g.b_router);
  F(g.w_mlp1); F(g.w_mlp2); F(g.b_mlp1); F(g.b_mlp2); F(g.rms_out_w); F(g.out);

  RunState &s = T->state;
  F(s.x); F(s.t); F(s.tb); F(s.tb2); F(s.router_score); if(s.topk_v) hipFree(s.topk_v);
  if(s.topk_i) hipFree(s.topk_i); F(s.mlp1_out); F(s.gate); F(s.up); F(s.gate_up); F(s.e_agg);
  F(s.qkv); F(s.q); F(s.att); F(s.logits); F(s.key_cache); F(s.value_cache);
  if (s.mask) hipFree(s.mask);
}

static void build_transformer_gpu(Transformer *T, char *ckpt) {
  T->fd = -1; T->data = nullptr; T->file_size = 0;
  hipSetDevice(0); // MI250 GCD0
  load_checkpoint_gpu(ckpt, &T->config, &T->weights, &T->fd, &T->data, &T->file_size);
  malloc_state_gpu(T);
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

  build_transformer_gpu(transformer, checkpoint_path);
  read_tokenizer(tokenizer, tokenizer_path, transformer->config.vocab_size);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  free_transformer_gpu(transformer);
  free_tokenizer(tokenizer);
}

static float* forward_gpu(Transformer *T, int token, int pos) {
  const Config &p = T->config;
  const TransformerWeights &w = T->weights;
  RunState &s = T->state;

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
    float *k_buf = T->state.key_cache + (size_t)l*p.seq_len*kv_dim + (size_t)pos*kv_dim;
    float *v_buf = T->state.value_cache + (size_t)l*p.seq_len*kv_dim + (size_t)pos*kv_dim;
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
    float *k_layer_cache = T->state.key_cache + (size_t)l*p.seq_len*kv_dim;
    float *v_layer_cache = T->state.value_cache + (size_t)l*p.seq_len*kv_dim;
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
    softmax_rows_gpu(s.att, Hq, pos + 2, p.seq_len + 1);

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
    gemv_gpu(s.router_score, s.t, Wr, H, p.n_experts);
    add_bias_gpu(s.router_score, Br, p.n_experts);
    HIP_CHECK(hipDeviceSynchronize()); // ensure router on device finished

    // Bring router to host → top-k selection on CPU (simple, correct)
    float *h_router = (float*)malloc(p.n_experts*sizeof(float));
    float *h_topk_v = (float*)malloc(p.experts_per_token*sizeof(float));
    int   *h_topk_i = (int*)  malloc(p.experts_per_token*sizeof(int));
    HIP_CHECK(hipMemcpy(h_router, s.router_score, p.n_experts*sizeof(float), hipMemcpyDeviceToHost));
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
    float *logits = forward_gpu(transformer, token, pos);

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

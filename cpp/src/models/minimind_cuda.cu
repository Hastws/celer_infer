// minimind_cuda.cu - CUDA GPU inference implementation
// ============================================================================
// Complete CUDA implementation of MiniMind inference for GPU acceleration.
// This is a standalone file that compiles with nvcc.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../third_party/nlohmann/json.hpp"

using json = nlohmann::json;

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define CUBLAS_CHECK(call) do { \
  cublasStatus_t status = call; \
  if (status != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

// ============================================================================
// Constants
// ============================================================================
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// ============================================================================
// CUDA Kernels
// ============================================================================

// RMS Normalization kernel
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t hidden, float eps) {
  
  const int row = blockIdx.x;
  const float* row_in = x + row * hidden;
  float* row_out = out + row * hidden;
  
  extern __shared__ float shared[];
  
  // Compute sum of squares
  float local_ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float val = row_in[i];
    local_ss += val * val;
  }
  shared[threadIdx.x] = local_ss;
  __syncthreads();
  
  // Block reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  float ms = shared[0] / static_cast<float>(hidden);
  float inv_rms = rsqrtf(ms + eps);
  
  // Normalize
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    row_out[i] = row_in[i] * inv_rms * weight[i];
  }
}

// SiLU activation kernel
__global__ void silu_kernel(float* __restrict__ x, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = x[idx];
    x[idx] = val / (1.0f + expf(-val));
  }
}

// Element-wise multiply kernel
__global__ void mul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}

// Element-wise add kernel
__global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// RoPE kernel - rewritten with 1D linear indexing for stability
__global__ void rope_kernel(
    float* __restrict__ q, float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int64_t B, int64_t S, int64_t H, int64_t KVH, int64_t D) {
  
  // Total elements to process for Q: B * S * H * (D/2) pairs
  // We process one pair (d, d+1) per thread
  int64_t total_q = B * S * H * (D / 2);
  int64_t total_k = B * S * KVH * (D / 2);
  
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Process Q
  if (idx < total_q) {
    int64_t pair = idx % (D / 2);
    int64_t h = (idx / (D / 2)) % H;
    int64_t s = (idx / (D / 2 * H)) % S;
    int64_t b = idx / (D / 2 * H * S);
    
    float cos_val = cos_cache[s * D + pair * 2];
    float sin_val = sin_cache[s * D + pair * 2];
    
    int64_t q_idx = ((b * S + s) * H + h) * D + pair * 2;
    float q0 = q[q_idx];
    float q1 = q[q_idx + 1];
    q[q_idx]     = q0 * cos_val - q1 * sin_val;
    q[q_idx + 1] = q0 * sin_val + q1 * cos_val;
  }
  
  // Process K (separate grid launch would be cleaner but this works)
  if (idx < total_k) {
    int64_t pair = idx % (D / 2);
    int64_t h = (idx / (D / 2)) % KVH;
    int64_t s = (idx / (D / 2 * KVH)) % S;
    int64_t b = idx / (D / 2 * KVH * S);
    
    float cos_val = cos_cache[s * D + pair * 2];
    float sin_val = sin_cache[s * D + pair * 2];
    
    int64_t k_idx = ((b * S + s) * KVH + h) * D + pair * 2;
    float k0 = k[k_idx];
    float k1 = k[k_idx + 1];
    k[k_idx]     = k0 * cos_val - k1 * sin_val;
    k[k_idx + 1] = k0 * sin_val + k1 * cos_val;
  }
}

// Repeat KV heads kernel
__global__ void repeat_kv_kernel(
    const float* __restrict__ kv_in,  // (B, S, KVH, D)
    float* __restrict__ kv_out,       // (B, S, H, D)
    int64_t B, int64_t S, int64_t KVH, int64_t H, int64_t D, int64_t n_rep) {
  
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * S * H * D;
  if (idx >= total) return;
  
  // Decode index
  int64_t d = idx % D;
  int64_t h = (idx / D) % H;
  int64_t s = (idx / (D * H)) % S;
  int64_t b = idx / (D * H * S);
  
  // Map to KVH
  int64_t kvh = h / n_rep;
  
  kv_out[idx] = kv_in[((b * S + s) * KVH + kvh) * D + d];
}

// Transpose BSHD -> BHSD kernel
__global__ void transpose_bshd_to_bhsd_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t B, int64_t S, int64_t H, int64_t D) {
  
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * H * S * D;
  if (idx >= total) return;
  
  int64_t d = idx % D;
  int64_t s = (idx / D) % S;
  int64_t h = (idx / (D * S)) % H;
  int64_t b = idx / (D * S * H);
  
  // in: (B, S, H, D), out: (B, H, S, D)
  int64_t in_idx = ((b * S + s) * H + h) * D + d;
  out[idx] = in[in_idx];
}

// Transpose BHSD -> BSHD kernel
__global__ void transpose_bhsd_to_bshd_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t B, int64_t H, int64_t S, int64_t D) {
  
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * S * H * D;
  if (idx >= total) return;
  
  int64_t d = idx % D;
  int64_t h = (idx / D) % H;
  int64_t s = (idx / (D * H)) % S;
  int64_t b = idx / (D * H * S);
  
  // in: (B, H, S, D), out: (B, S, H, D)
  int64_t in_idx = ((b * H + h) * S + s) * D + d;
  out[idx] = in[in_idx];
}

// Causal softmax kernel (fused scale + mask + softmax)
__global__ void causal_softmax_kernel(
    float* __restrict__ scores,
    int64_t B, int64_t H, int64_t S, int64_t T,
    float scale) {
  
  const int row = blockIdx.x;
  const int bh = row / S;
  const int sq = row % S;
  
  float* row_data = scores + row * T;
  
  extern __shared__ float shared[];
  
  const float neg_inf = -1e9f;
  
  // Scale and mask, find max
  float local_max = neg_inf;
  for (int sk = threadIdx.x; sk < T; sk += blockDim.x) {
    float val = row_data[sk] * scale;
    if (sk > sq) val = neg_inf;  // Causal mask
    row_data[sk] = val;
    local_max = fmaxf(local_max, val);
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  
  // Reduce max
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float max_val = shared[0];
  
  // Exp and sum
  float local_sum = 0.0f;
  for (int sk = threadIdx.x; sk < T; sk += blockDim.x) {
    float exp_val = expf(row_data[sk] - max_val);
    row_data[sk] = exp_val;
    local_sum += exp_val;
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float inv_sum = 1.0f / shared[0];
  
  // Normalize
  for (int sk = threadIdx.x; sk < T; sk += blockDim.x) {
    row_data[sk] *= inv_sum;
  }
}

// Embedding lookup kernel
__global__ void embedding_kernel(
    const float* __restrict__ table,
    const int32_t* __restrict__ indices,
    float* __restrict__ out,
    int64_t batch_seq, int64_t hidden) {
  
  const int row = blockIdx.x;
  const int64_t idx = indices[row];
  const float* src = table + idx * hidden;
  float* dst = out + row * hidden;
  
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// ============================================================================
// Model Configuration
// ============================================================================
struct minimind_config {
  int64_t vocab_size = 0;
  int64_t hidden = 0;
  int64_t n_layers = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t head_dim = 0;
  int64_t inter = 0;
  float rms_eps = 1e-5f;
  int64_t max_pos = 0;
};

// ============================================================================
// CUDA Inference Context
// ============================================================================
struct CudaContext {
  cublasHandle_t cublas;
  
  // Device buffers
  float* d_h0 = nullptr;
  float* d_h1 = nullptr;
  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_k_rep = nullptr;
  float* d_v_rep = nullptr;
  float* d_q_bhsd = nullptr;
  float* d_k_bhsd = nullptr;
  float* d_v_bhsd = nullptr;
  float* d_scores = nullptr;
  float* d_attn_out = nullptr;
  float* d_attn_out_bshd = nullptr;
  float* d_ffn_gate = nullptr;
  float* d_ffn_up = nullptr;
  float* d_ffn_mid = nullptr;
  float* d_ffn_out = nullptr;
  float* d_logits = nullptr;
  
  // Weights on device
  float* d_tok_emb = nullptr;
  float* d_final_rms = nullptr;
  float* d_lm_head = nullptr;
  float* d_rope_cos = nullptr;
  float* d_rope_sin = nullptr;
  
  std::vector<float*> d_rms_attn;
  std::vector<float*> d_rms_ffn;
  std::vector<float*> d_wq;
  std::vector<float*> d_wk;
  std::vector<float*> d_wv;
  std::vector<float*> d_wo;
  std::vector<float*> d_wgate;
  std::vector<float*> d_wup;
  std::vector<float*> d_wdown;
  
  int32_t* d_input_ids = nullptr;
  
  void init(const minimind_config& cfg, int64_t B, int64_t S) {
    CUBLAS_CHECK(cublasCreate(&cublas));
    
    const int64_t H = cfg.n_heads;
    const int64_t KVH = cfg.n_kv_heads;
    const int64_t D = cfg.head_dim;
    const int64_t T = S;
    
    // Allocate workspace
    CUDA_CHECK(cudaMalloc(&d_h0, B * S * cfg.hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h1, B * S * cfg.hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, B * S * H * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, B * S * KVH * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, B * S * KVH * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_rep, B * T * H * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_rep, B * T * H * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_bhsd, B * H * S * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_bhsd, B * H * T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_bhsd, B * H * T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, B * H * S * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out, B * H * S * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out_bshd, B * S * H * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_gate, B * S * cfg.inter * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_up, B * S * cfg.inter * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_mid, B * S * cfg.inter * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_out, B * S * cfg.hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, B * S * cfg.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_ids, B * S * sizeof(int32_t)));
  }
  
  void upload_weights(const minimind_config& cfg,
                      const float* tok_emb, const float* final_rms, const float* lm_head,
                      const float* rope_cos, const float* rope_sin,
                      const std::vector<std::vector<float>>& rms_attn,
                      const std::vector<std::vector<float>>& rms_ffn,
                      const std::vector<std::vector<float>>& wq,
                      const std::vector<std::vector<float>>& wk,
                      const std::vector<std::vector<float>>& wv,
                      const std::vector<std::vector<float>>& wo,
                      const std::vector<std::vector<float>>& wgate,
                      const std::vector<std::vector<float>>& wup,
                      const std::vector<std::vector<float>>& wdown) {
    
    // Token embedding
    CUDA_CHECK(cudaMalloc(&d_tok_emb, cfg.vocab_size * cfg.hidden * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tok_emb, tok_emb, cfg.vocab_size * cfg.hidden * sizeof(float), cudaMemcpyHostToDevice));
    
    // Final RMS
    CUDA_CHECK(cudaMalloc(&d_final_rms, cfg.hidden * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_final_rms, final_rms, cfg.hidden * sizeof(float), cudaMemcpyHostToDevice));
    
    // LM head
    CUDA_CHECK(cudaMalloc(&d_lm_head, cfg.vocab_size * cfg.hidden * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lm_head, lm_head, cfg.vocab_size * cfg.hidden * sizeof(float), cudaMemcpyHostToDevice));
    
    // RoPE
    CUDA_CHECK(cudaMalloc(&d_rope_cos, cfg.max_pos * cfg.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rope_cos, rope_cos, cfg.max_pos * cfg.head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_rope_sin, cfg.max_pos * cfg.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rope_sin, rope_sin, cfg.max_pos * cfg.head_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Layer weights
    d_rms_attn.resize(cfg.n_layers);
    d_rms_ffn.resize(cfg.n_layers);
    d_wq.resize(cfg.n_layers);
    d_wk.resize(cfg.n_layers);
    d_wv.resize(cfg.n_layers);
    d_wo.resize(cfg.n_layers);
    d_wgate.resize(cfg.n_layers);
    d_wup.resize(cfg.n_layers);
    d_wdown.resize(cfg.n_layers);
    
    for (int64_t l = 0; l < cfg.n_layers; ++l) {
      CUDA_CHECK(cudaMalloc(&d_rms_attn[l], cfg.hidden * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_rms_attn[l], rms_attn[l].data(), cfg.hidden * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CHECK(cudaMalloc(&d_rms_ffn[l], cfg.hidden * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_rms_ffn[l], rms_ffn[l].data(), cfg.hidden * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CHECK(cudaMalloc(&d_wq[l], cfg.n_heads * cfg.head_dim * cfg.hidden * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_wq[l], wq[l].data(), wq[l].size() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CHECK(cudaMalloc(&d_wk[l], cfg.n_kv_heads * cfg.head_dim * cfg.hidden * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_wk[l], wk[l].data(), wk[l].size() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CHECK(cudaMalloc(&d_wv[l], cfg.n_kv_heads * cfg.head_dim * cfg.hidden * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_wv[l], wv[l].data(), wv[l].size() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CHECK(cudaMalloc(&d_wo[l], cfg.hidden * cfg.n_heads * cfg.head_dim * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_wo[l], wo[l].data(), wo[l].size() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CHECK(cudaMalloc(&d_wgate[l], cfg.inter * cfg.hidden * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_wgate[l], wgate[l].data(), wgate[l].size() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CHECK(cudaMalloc(&d_wup[l], cfg.inter * cfg.hidden * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_wup[l], wup[l].data(), wup[l].size() * sizeof(float), cudaMemcpyHostToDevice));
      
      CUDA_CHECK(cudaMalloc(&d_wdown[l], cfg.hidden * cfg.inter * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_wdown[l], wdown[l].data(), wdown[l].size() * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  
  ~CudaContext() {
    cublasDestroy(cublas);
    // Note: In production, free all allocated memory
  }
};

// ============================================================================
// cuBLAS GEMM wrapper: C = A @ B^T
// ============================================================================
void cublas_matmul_nt(cublasHandle_t handle,
                      const float* A, int64_t M, int64_t K,
                      const float* B, int64_t N,
                      float* C) {
  float alpha = 1.0f, beta = 0.0f;
  // cuBLAS is column-major, so we compute C^T = B @ A^T
  CUBLAS_CHECK(cublasSgemm(handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, K,
      A, K,
      &beta,
      C, N));
}

// ============================================================================
// Forward pass
// ============================================================================
void minimind_forward_cuda(
    const minimind_config& cfg,
    CudaContext& ctx,
    const int32_t* input_ids, int64_t B, int64_t S) {
  
  const int64_t H = cfg.n_heads;
  const int64_t KVH = cfg.n_kv_heads;
  const int64_t D = cfg.head_dim;
  const int64_t T = S;
  const float scale = 1.0f / sqrtf(static_cast<float>(D));
  
  // Copy input to device
  CUDA_CHECK(cudaMemcpy(ctx.d_input_ids, input_ids, B * S * sizeof(int32_t), cudaMemcpyHostToDevice));
  
  // Embedding lookup
  embedding_kernel<<<B * S, std::min((int64_t)BLOCK_SIZE, cfg.hidden)>>>(
      ctx.d_tok_emb, ctx.d_input_ids, ctx.d_h0, B * S, cfg.hidden);
  
  // Layer stack
  for (int64_t l = 0; l < cfg.n_layers; ++l) {
    // Pre-attn RMSNorm
    rms_norm_kernel<<<B * S, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        ctx.d_h0, ctx.d_rms_attn[l], ctx.d_h1, cfg.hidden, cfg.rms_eps);
    
    // Q, K, V projections
    cublas_matmul_nt(ctx.cublas, ctx.d_h1, B * S, cfg.hidden, ctx.d_wq[l], H * D, ctx.d_q);
    cublas_matmul_nt(ctx.cublas, ctx.d_h1, B * S, cfg.hidden, ctx.d_wk[l], KVH * D, ctx.d_k);
    cublas_matmul_nt(ctx.cublas, ctx.d_h1, B * S, cfg.hidden, ctx.d_wv[l], KVH * D, ctx.d_v);
    
    // RoPE - use 1D grid
    int64_t rope_total = std::max(B * S * H * (D / 2), B * S * KVH * (D / 2));
    rope_kernel<<<(rope_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_q, ctx.d_k, ctx.d_rope_cos, ctx.d_rope_sin, B, S, H, KVH, D);
    
    // Repeat KV
    int64_t n_rep = H / KVH;
    int kv_total = B * T * H * D;
    repeat_kv_kernel<<<(kv_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_k, ctx.d_k_rep, B, T, KVH, H, D, n_rep);
    repeat_kv_kernel<<<(kv_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_v, ctx.d_v_rep, B, T, KVH, H, D, n_rep);
    
    // Transpose Q: (B,S,H,D) -> (B,H,S,D)
    int total_qkv = B * H * S * D;
    transpose_bshd_to_bhsd_kernel<<<(total_qkv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_q, ctx.d_q_bhsd, B, S, H, D);
    transpose_bshd_to_bhsd_kernel<<<(total_qkv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_k_rep, ctx.d_k_bhsd, B, T, H, D);
    transpose_bshd_to_bhsd_kernel<<<(total_qkv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_v_rep, ctx.d_v_bhsd, B, T, H, D);
    
    // Attention: scores = Q @ K^T (batched)
    // Q: (B, H, S, D), K: (B, H, T, D), scores: (B, H, S, T)
    // For batched gemm, we treat B*H as the batch dimension
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(ctx.cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        T, S, D,
        &alpha,
        ctx.d_k_bhsd, D, T * D,  // strideA = T * D (per-head stride)
        ctx.d_q_bhsd, D, S * D,  // strideB = S * D (per-head stride)
        &beta,
        ctx.d_scores, T, S * T,  // strideC = S * T (per-head stride)
        B * H));
    
    // Causal softmax
    int total_rows = B * H * S;
    causal_softmax_kernel<<<total_rows, std::min((int64_t)BLOCK_SIZE, T), BLOCK_SIZE * sizeof(float)>>>(
        ctx.d_scores, B, H, S, T, scale);
    
    // Attn @ V
    // scores: (B, H, S, T), V: (B, H, T, D), out: (B, H, S, D)
    CUBLAS_CHECK(cublasSgemmStridedBatched(ctx.cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, S, T,
        &alpha,
        ctx.d_v_bhsd, D, T * D,     // strideA = T * D
        ctx.d_scores, T, S * T,     // strideB = S * T
        &beta,
        ctx.d_attn_out, D, S * D,   // strideC = S * D
        B * H));
    
    // Transpose back: (B,H,S,D) -> (B,S,H,D)
    transpose_bhsd_to_bshd_kernel<<<(total_qkv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_attn_out, ctx.d_attn_out_bshd, B, H, S, D);
    
    // O projection
    cublas_matmul_nt(ctx.cublas, ctx.d_attn_out_bshd, B * S, H * D, ctx.d_wo[l], cfg.hidden, ctx.d_h1);
    
    // Residual
    int total = B * S * cfg.hidden;
    add_kernel<<<(total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_h0, ctx.d_h1, ctx.d_h0, total);
    
    // FFN
    // Pre-FFN RMSNorm
    rms_norm_kernel<<<B * S, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        ctx.d_h0, ctx.d_rms_ffn[l], ctx.d_h1, cfg.hidden, cfg.rms_eps);
    
    // Gate and Up projections
    cublas_matmul_nt(ctx.cublas, ctx.d_h1, B * S, cfg.hidden, ctx.d_wgate[l], cfg.inter, ctx.d_ffn_gate);
    cublas_matmul_nt(ctx.cublas, ctx.d_h1, B * S, cfg.hidden, ctx.d_wup[l], cfg.inter, ctx.d_ffn_up);
    
    // SiLU(gate) * up
    int ffn_total = B * S * cfg.inter;
    silu_kernel<<<(ffn_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(ctx.d_ffn_gate, ffn_total);
    mul_kernel<<<(ffn_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_ffn_gate, ctx.d_ffn_up, ctx.d_ffn_mid, ffn_total);
    
    // Down projection
    cublas_matmul_nt(ctx.cublas, ctx.d_ffn_mid, B * S, cfg.inter, ctx.d_wdown[l], cfg.hidden, ctx.d_ffn_out);
    
    // Residual
    add_kernel<<<(total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.d_h0, ctx.d_ffn_out, ctx.d_h0, total);
  }
  
  // Final RMSNorm
  rms_norm_kernel<<<B * S, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
      ctx.d_h0, ctx.d_final_rms, ctx.d_h1, cfg.hidden, cfg.rms_eps);
  
  // LM head
  cublas_matmul_nt(ctx.cublas, ctx.d_h1, B * S, cfg.hidden, ctx.d_lm_head, cfg.vocab_size, ctx.d_logits);
}

// ============================================================================
// JSON utilities
// ============================================================================
static inline int b64_char_to_val(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

static std::vector<uint8_t> b64_decode(const std::string& b64) {
  std::vector<uint8_t> ret;
  int val = 0, bits = 0;
  for (unsigned char c : b64) {
    int digit = b64_char_to_val(c);
    if (digit == -1) continue;
    val = (val << 6) | digit;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      ret.push_back((val >> bits) & 0xFF);
    }
  }
  return ret;
}

static std::vector<float> load_tensor_from_json(const json& j) {
  std::string b64_data = j["data"];
  std::vector<uint8_t> decoded = b64_decode(b64_data);
  std::vector<float> result(decoded.size() / sizeof(float));
  std::memcpy(result.data(), decoded.data(), decoded.size());
  return result;
}

static void write_f32(const std::string& path, const float* x, size_t n) {
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(x), n * sizeof(float));
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
  std::string json_path = (argc >= 2) ? argv[1] : "dump_minimind/minimind.json";
  std::string dump_dir = (argc >= 3) ? argv[2] : "dump_minimind";
  
  std::cout << "=== MiniMind CUDA Inference ===\n\n";
  
  // Print GPU info
  int device;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU: " << prop.name << " (" << prop.totalGlobalMem / (1024*1024) << " MB)\n\n";
  
  // Load JSON
  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    std::cerr << "Failed to open: " << json_path << "\n";
    return 1;
  }
  json j;
  json_file >> j;
  json_file.close();
  
  // Parse config
  auto cfg_j = j["config"];
  minimind_config cfg;
  cfg.vocab_size = cfg_j["vocab_size"];
  cfg.hidden = cfg_j["hidden_size"];
  cfg.n_layers = cfg_j["num_hidden_layers"];
  cfg.n_heads = cfg_j["num_attention_heads"];
  cfg.n_kv_heads = cfg_j["num_key_value_heads"];
  cfg.inter = cfg_j["intermediate_size"];
  cfg.max_pos = cfg_j["max_position_embeddings"];
  cfg.head_dim = cfg.hidden / cfg.n_heads;
  
  auto meta = j["meta"];
  const int64_t B = meta["B"];
  const int64_t S = meta["S"];
  
  std::cout << "Config: hidden=" << cfg.hidden << " layers=" << cfg.n_layers
            << " heads=" << cfg.n_heads << " vocab=" << cfg.vocab_size << "\n";
  std::cout << "Input: B=" << B << " S=" << S << "\n\n";
  
  // Load inputs
  std::string input_ids_b64 = j["inputs"]["input_ids"]["data"];
  std::vector<uint8_t> input_ids_bytes = b64_decode(input_ids_b64);
  std::vector<int32_t> input_ids(input_ids_bytes.size() / sizeof(int32_t));
  std::memcpy(input_ids.data(), input_ids_bytes.data(), input_ids_bytes.size());
  
  // Load weights
  std::vector<float> rope_cos = load_tensor_from_json(j["rope"]["cos"]);
  std::vector<float> rope_sin = load_tensor_from_json(j["rope"]["sin"]);
  std::vector<float> tok_embedding = load_tensor_from_json(j["weights"]["tok_embedding"]);
  std::vector<float> final_rms = load_tensor_from_json(j["weights"]["final_rms"]);
  std::vector<float> lm_head = load_tensor_from_json(j["weights"]["lm_head"]);
  
  std::vector<std::vector<float>> rms_attn(cfg.n_layers), rms_ffn(cfg.n_layers);
  std::vector<std::vector<float>> wq(cfg.n_layers), wk(cfg.n_layers), wv(cfg.n_layers), wo(cfg.n_layers);
  std::vector<std::vector<float>> wgate(cfg.n_layers), wup(cfg.n_layers), wdown(cfg.n_layers);
  
  auto layers_j = j["weights"]["layers"];
  for (int64_t l = 0; l < cfg.n_layers; ++l) {
    auto layer_j = layers_j[l];
    rms_attn[l] = load_tensor_from_json(layer_j["rms_attn"]);
    rms_ffn[l] = load_tensor_from_json(layer_j["rms_ffn"]);
    wq[l] = load_tensor_from_json(layer_j["wq"]);
    wk[l] = load_tensor_from_json(layer_j["wk"]);
    wv[l] = load_tensor_from_json(layer_j["wv"]);
    wo[l] = load_tensor_from_json(layer_j["wo"]);
    wgate[l] = load_tensor_from_json(layer_j["w_gate"]);
    wup[l] = load_tensor_from_json(layer_j["w_up"]);
    wdown[l] = load_tensor_from_json(layer_j["w_down"]);
  }
  
  // Initialize CUDA context
  CudaContext ctx;
  ctx.init(cfg, B, S);
  ctx.upload_weights(cfg, tok_embedding.data(), final_rms.data(), lm_head.data(),
                     rope_cos.data(), rope_sin.data(),
                     rms_attn, rms_ffn, wq, wk, wv, wo, wgate, wup, wdown);
  
  // Warmup
  minimind_forward_cuda(cfg, ctx, input_ids.data(), B, S);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Timed runs
  const int num_runs = 100;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; ++i) {
    minimind_forward_cuda(cfg, ctx, input_ids.data(), B, S);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
  
  // Copy results back
  std::vector<float> logits(B * S * cfg.vocab_size);
  CUDA_CHECK(cudaMemcpy(logits.data(), ctx.d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
  
  // Stats
  float min_logit = logits[0], max_logit = logits[0];
  double mean_logit = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    min_logit = std::min(min_logit, logits[i]);
    max_logit = std::max(max_logit, logits[i]);
    mean_logit += logits[i];
  }
  mean_logit /= logits.size();
  
  std::cout << "========================================\n";
  std::cout << "[Forward] Shape: (" << B << ", " << S << ", " << cfg.vocab_size << ")\n";
  std::cout << "[Backend] CUDA (cuBLAS)\n";
  std::cout << "[Timing] " << elapsed_ms << " ms (avg of " << num_runs << " runs)\n";
  std::cout << "[Logits] Min: " << min_logit << ", Max: " << max_logit
            << ", Mean: " << mean_logit << "\n";
  std::cout << "========================================\n";
  
  // Save
  std::string logits_path = dump_dir + "/logits_cuda.npy";
  write_f32(logits_path, logits.data(), logits.size());
  std::cout << "[OK] Saved: " << logits_path << "\n";
  
  return 0;
}

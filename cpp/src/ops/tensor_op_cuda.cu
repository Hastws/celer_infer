// tensor_op_cuda.cu - CUDA kernel implementations
// ============================================================================

#include "tensor_op_cuda.cuh"

#ifdef USE_CUDA

namespace cuda {

// ============================================================================
// Constants
// ============================================================================
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// ============================================================================
// RMS Normalization Kernel
// ============================================================================
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t hidden, float eps) {
  
  const int row = blockIdx.x;
  const float* row_in = x + row * hidden;
  float* row_out = out + row * hidden;
  
  // Compute mean of squares using parallel reduction
  __shared__ float shared_ms[BLOCK_SIZE];
  
  float local_ms = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float val = row_in[i];
    local_ms += val * val;
  }
  shared_ms[threadIdx.x] = local_ms;
  __syncthreads();
  
  // Block reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_ms[threadIdx.x] += shared_ms[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  float ms = shared_ms[0] / static_cast<float>(hidden);
  float inv_rms = rsqrtf(ms + eps);
  
  // Normalize
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    row_out[i] = row_in[i] * inv_rms * weight[i];
  }
}

void cuda_rms_norm(
    const float* x, int64_t batch_seq, int64_t hidden,
    const float* weight, float eps,
    float* out, cudaStream_t stream) {
  
  dim3 grid(batch_seq);
  dim3 block(min(static_cast<int64_t>(BLOCK_SIZE), hidden));
  
  rms_norm_kernel<<<grid, block, 0, stream>>>(x, weight, out, hidden, eps);
}

// ============================================================================
// RoPE Kernel
// ============================================================================
__global__ void rope_kernel(
    float* __restrict__ q, float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int64_t B, int64_t S, int64_t H, int64_t D,
    int64_t past_len, int64_t max_len) {
  
  // Each thread handles one (head, position, pair) combination
  const int64_t b = blockIdx.z;
  const int64_t h = blockIdx.y;
  const int64_t s = blockIdx.x;
  const int64_t d = threadIdx.x;  // 0..D/2-1
  
  if (d >= D / 2) return;
  
  const int64_t pos = past_len + s;
  const int64_t cache_idx = pos * D + d * 2;
  
  float cos_val = cos_cache[cache_idx];
  float sin_val = sin_cache[cache_idx];
  
  // Q
  int64_t q_idx = ((b * H + h) * S + s) * D + d * 2;
  float q0 = q[q_idx];
  float q1 = q[q_idx + 1];
  q[q_idx]     = q0 * cos_val - q1 * sin_val;
  q[q_idx + 1] = q0 * sin_val + q1 * cos_val;
  
  // K (only for KV heads, but for simplicity apply to all)
  int64_t k_idx = ((b * H + h) * S + s) * D + d * 2;
  float k0 = k[k_idx];
  float k1 = k[k_idx + 1];
  k[k_idx]     = k0 * cos_val - k1 * sin_val;
  k[k_idx + 1] = k0 * sin_val + k1 * cos_val;
}

void cuda_apply_rope(
    float* q, float* k,
    int64_t B, int64_t S, int64_t H, int64_t D,
    const float* cos_cache, const float* sin_cache,
    int64_t past_len, int64_t max_len,
    cudaStream_t stream) {
  
  dim3 grid(S, H, B);
  dim3 block(D / 2);
  
  rope_kernel<<<grid, block, 0, stream>>>(
      q, k, cos_cache, sin_cache, B, S, H, D, past_len, max_len);
}

// ============================================================================
// SiLU Kernel
// ============================================================================
__global__ void silu_kernel(float* __restrict__ x, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = x[idx];
    x[idx] = val / (1.0f + expf(-val));
  }
}

void cuda_silu(float* x, int64_t n, cudaStream_t stream) {
  int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  silu_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(x, n);
}

// ============================================================================
// Element-wise operations
// ============================================================================
__global__ void mul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}

void cuda_mul(const float* a, const float* b, float* c, int64_t n, cudaStream_t stream) {
  int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  mul_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, n);
}

__global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void cuda_add(const float* a, const float* b, float* c, int64_t n, cudaStream_t stream) {
  int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  add_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, n);
}

// ============================================================================
// Softmax Kernel (one block per row for small dim)
// ============================================================================
__global__ void softmax_kernel(
    float* __restrict__ x,
    int64_t batch, int64_t dim) {
  
  const int row = blockIdx.x;
  float* row_data = x + row * dim;
  
  __shared__ float shared[BLOCK_SIZE];
  
  // Find max
  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    local_max = fmaxf(local_max, row_data[i]);
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float max_val = shared[0];
  
  // Compute exp and sum
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float exp_val = expf(row_data[i] - max_val);
    row_data[i] = exp_val;
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
  float sum_val = shared[0];
  
  // Normalize
  float inv_sum = 1.0f / sum_val;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    row_data[i] *= inv_sum;
  }
}

void cuda_softmax(float* x, int64_t batch, int64_t dim, cudaStream_t stream) {
  dim3 grid(batch);
  dim3 block(min(static_cast<int64_t>(BLOCK_SIZE), dim));
  softmax_kernel<<<grid, block, 0, stream>>>(x, batch, dim);
}

// ============================================================================
// Causal Softmax (fused scale + mask + softmax)
// ============================================================================
__global__ void causal_softmax_kernel(
    float* __restrict__ scores,
    int64_t B, int64_t H, int64_t S_q, int64_t S_k,
    int64_t past_len, float scale) {
  
  // Each block handles one query position
  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int sq = blockIdx.x;
  
  float* row = scores + ((b * H + h) * S_q + sq) * S_k;
  
  __shared__ float shared[BLOCK_SIZE];
  
  const float neg_inf = -1e9f;
  const int64_t query_pos = past_len + sq;
  
  // Scale and mask, find max
  float local_max = neg_inf;
  for (int sk = threadIdx.x; sk < S_k; sk += blockDim.x) {
    float val = row[sk] * scale;
    if (sk > query_pos) val = neg_inf;  // Causal mask
    row[sk] = val;
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
  for (int sk = threadIdx.x; sk < S_k; sk += blockDim.x) {
    float exp_val = expf(row[sk] - max_val);
    row[sk] = exp_val;
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
  for (int sk = threadIdx.x; sk < S_k; sk += blockDim.x) {
    row[sk] *= inv_sum;
  }
}

void cuda_causal_softmax(
    float* scores,
    int64_t B, int64_t H, int64_t S_q, int64_t S_k,
    int64_t past_len, float scale,
    cudaStream_t stream) {
  
  dim3 grid(S_q, H, B);
  dim3 block(min(static_cast<int64_t>(BLOCK_SIZE), S_k));
  
  causal_softmax_kernel<<<grid, block, 0, stream>>>(
      scores, B, H, S_q, S_k, past_len, scale);
}

// ============================================================================
// Embedding Lookup
// ============================================================================
__global__ void embedding_kernel(
    const float* __restrict__ table,
    const int64_t* __restrict__ indices,
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

void cuda_embedding(
    const float* table, const int64_t* indices,
    int64_t batch_seq, int64_t vocab, int64_t hidden,
    float* out, cudaStream_t stream) {
  
  dim3 grid(batch_seq);
  dim3 block(min(static_cast<int64_t>(BLOCK_SIZE), hidden));
  
  embedding_kernel<<<grid, block, 0, stream>>>(table, indices, out, batch_seq, hidden);
}

}  // namespace cuda

#endif  // USE_CUDA

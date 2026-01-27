// tensor_op_cuda.cuh - CUDA GPU tensor operations header
// ============================================================================
// CUDA implementation of tensor operations for GPU inference:
//   - GPU memory management utilities
//   - CUDA kernel declarations
//   - cuBLAS integration for GEMM
// ============================================================================

#ifndef TENSOR_OP_CUDA_CUH
#define TENSOR_OP_CUDA_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <memory>

namespace cuda {

// ============================================================================
// CUDA error checking macros
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
// GPU Memory Manager: RAII wrapper for CUDA allocations
// ============================================================================
template<typename T>
class CudaBuffer {
public:
  CudaBuffer() : data_(nullptr), size_(0) {}
  
  explicit CudaBuffer(int64_t n) : size_(n) {
    CUDA_CHECK(cudaMalloc(&data_, n * sizeof(T)));
  }
  
  ~CudaBuffer() {
    if (data_) cudaFree(data_);
  }
  
  // Move semantics
  CudaBuffer(CudaBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  
  CudaBuffer& operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
      if (data_) cudaFree(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  
  // Disable copy
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  
  T* data() { return data_; }
  const T* data() const { return data_; }
  int64_t size() const { return size_; }
  
  void resize(int64_t n) {
    if (n != size_) {
      if (data_) cudaFree(data_);
      size_ = n;
      CUDA_CHECK(cudaMalloc(&data_, n * sizeof(T)));
    }
  }
  
  // Copy from host
  void copy_from_host(const T* host_data, int64_t n) {
    if (n > size_) resize(n);
    CUDA_CHECK(cudaMemcpy(data_, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  
  // Copy to host
  void copy_to_host(T* host_data, int64_t n) const {
    CUDA_CHECK(cudaMemcpy(host_data, data_, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
  
private:
  T* data_;
  int64_t size_;
};

// ============================================================================
// cuBLAS handle singleton
// ============================================================================
class CublasHandle {
public:
  static CublasHandle& instance() {
    static CublasHandle inst;
    return inst;
  }
  
  cublasHandle_t handle() { return handle_; }
  
private:
  CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
  ~CublasHandle() { cublasDestroy(handle_); }
  cublasHandle_t handle_;
};

// ============================================================================
// CUDA Kernel Declarations (implementations in .cu file)
// ============================================================================

// RMS Normalization kernel
void cuda_rms_norm(
    const float* x, int64_t batch_seq, int64_t hidden,
    const float* weight, float eps,
    float* out, cudaStream_t stream = 0);

// RoPE (Rotary Position Embedding) kernel
void cuda_apply_rope(
    float* q, float* k,
    int64_t B, int64_t S, int64_t H, int64_t D,
    const float* cos_cache, const float* sin_cache,
    int64_t past_len, int64_t max_len,
    cudaStream_t stream = 0);

// SiLU activation kernel
void cuda_silu(float* x, int64_t n, cudaStream_t stream = 0);

// Element-wise multiply kernel
void cuda_mul(const float* a, const float* b, float* c, int64_t n, cudaStream_t stream = 0);

// Element-wise add kernel
void cuda_add(const float* a, const float* b, float* c, int64_t n, cudaStream_t stream = 0);

// Softmax kernel
void cuda_softmax(float* x, int64_t batch, int64_t dim, cudaStream_t stream = 0);

// Causal mask + softmax (fused)
void cuda_causal_softmax(
    float* scores,
    int64_t B, int64_t H, int64_t S_q, int64_t S_k,
    int64_t past_len, float scale,
    cudaStream_t stream = 0);

// Embedding lookup
void cuda_embedding(
    const float* table, const int64_t* indices,
    int64_t batch_seq, int64_t vocab, int64_t hidden,
    float* out, cudaStream_t stream = 0);

// ============================================================================
// High-level CUDA operations using cuBLAS
// ============================================================================

// Matrix multiplication: C = A @ B^T
// A: (M, K), B: (N, K), C: (M, N)
inline void cuda_matmul(
    cublasHandle_t handle,
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    bool transA = false, bool transB = true) {
  
  float alpha = 1.0f, beta = 0.0f;
  
  // cuBLAS is column-major, so we compute C^T = B^T @ A^T
  // When transB=true: C = A @ B^T becomes C^T = B @ A^T
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
  
  // Dimensions for column-major
  int lda = transA ? M : K;
  int ldb = transB ? K : N;
  int ldc = N;
  
  CUBLAS_CHECK(cublasSgemm(handle,
      opB, opA,
      N, M, K,
      &alpha,
      B, ldb,
      A, lda,
      &beta,
      C, ldc));
}

// Batched matrix multiplication
inline void cuda_batched_matmul(
    cublasHandle_t handle,
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K, int64_t batch,
    int64_t strideA, int64_t strideB, int64_t strideC,
    bool transA = false, bool transB = true) {
  
  float alpha = 1.0f, beta = 0.0f;
  
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
  
  int lda = transA ? M : K;
  int ldb = transB ? K : N;
  int ldc = N;
  
  CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
      opB, opA,
      N, M, K,
      &alpha,
      B, ldb, strideB,
      A, lda, strideA,
      &beta,
      C, ldc, strideC,
      batch));
}

// ============================================================================
// CUDA Inference Context: manages GPU resources for a model
// ============================================================================
struct CudaInferenceContext {
  CublasHandle& cublas;
  cudaStream_t stream;
  
  // Pre-allocated buffers
  CudaBuffer<float> hidden_buf1;
  CudaBuffer<float> hidden_buf2;
  CudaBuffer<float> attn_buf;
  CudaBuffer<float> ffn_buf;
  
  CudaInferenceContext() : cublas(CublasHandle::instance()), stream(0) {}
  
  void allocate_buffers(int64_t max_batch_seq, int64_t hidden, int64_t inter, int64_t heads, int64_t max_seq) {
    hidden_buf1.resize(max_batch_seq * hidden);
    hidden_buf2.resize(max_batch_seq * hidden);
    attn_buf.resize(max_batch_seq * heads * max_seq);
    ffn_buf.resize(max_batch_seq * inter);
  }
  
  void sync() {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
};

}  // namespace cuda

#else  // !USE_CUDA

// Stub definitions when CUDA is not available
namespace cuda {

struct CudaInferenceContext {
  void allocate_buffers(int64_t, int64_t, int64_t, int64_t, int64_t) {}
  void sync() {}
};

}  // namespace cuda

#endif  // USE_CUDA

#endif  // TENSOR_OP_CUDA_CUH

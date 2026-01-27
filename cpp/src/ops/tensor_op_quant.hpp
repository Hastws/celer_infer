// tensor_op_quant.hpp - Quantized tensor operations for INT8 inference
// ============================================================================
// Provides INT8 quantization for reduced memory bandwidth and faster compute:
//   - Symmetric per-tensor quantization
//   - Per-channel quantization for weights
//   - INT8 GEMM with INT32 accumulation
//   - Mixed-precision (INT8 weights, FP32 activations)
// ============================================================================

#ifndef TENSOR_OP_QUANT_HPP
#define TENSOR_OP_QUANT_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace quant {

// ============================================================================
// Quantization parameters
// ============================================================================
struct QuantParams {
  float scale;       // scale factor: float_value = int8_value * scale
  int32_t zero_point;  // zero point (0 for symmetric)
  
  QuantParams() : scale(1.0f), zero_point(0) {}
  QuantParams(float s, int32_t zp = 0) : scale(s), zero_point(zp) {}
};

// Per-channel quantization parameters (for weights)
struct PerChannelQuantParams {
  std::vector<float> scales;     // one scale per output channel
  std::vector<int32_t> zero_points;
  
  PerChannelQuantParams() = default;
  PerChannelQuantParams(int64_t channels) : scales(channels), zero_points(channels, 0) {}
};

// ============================================================================
// Symmetric quantization: find scale from absmax
// ============================================================================
inline QuantParams compute_symmetric_quant_params(
    const float* data, int64_t n, bool per_tensor = true) {
  
  float absmax = 0.0f;
  
#ifdef __AVX2__
  __m256 vmax = _mm256_setzero_ps();
  __m256 sign_mask = _mm256_set1_ps(-0.0f);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(data + i);
    __m256 vabs = _mm256_andnot_ps(sign_mask, v);  // abs
    vmax = _mm256_max_ps(vmax, vabs);
  }
  // Horizontal max
  __m128 lo = _mm256_castps256_ps128(vmax);
  __m128 hi = _mm256_extractf128_ps(vmax, 1);
  lo = _mm_max_ps(lo, hi);
  lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
  lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
  absmax = _mm_cvtss_f32(lo);
  for (; i < n; ++i) absmax = std::max(absmax, std::abs(data[i]));
#else
  for (int64_t i = 0; i < n; ++i) absmax = std::max(absmax, std::abs(data[i]));
#endif
  
  // INT8 symmetric range: [-127, 127] (avoiding -128 for symmetry)
  float scale = absmax / 127.0f;
  if (scale == 0.0f) scale = 1.0f;  // Handle all-zero case
  
  return QuantParams(scale, 0);
}

// Per-channel quantization for weight matrix (out_features, in_features)
inline PerChannelQuantParams compute_per_channel_quant_params(
    const float* weight, int64_t out_features, int64_t in_features) {
  
  PerChannelQuantParams params(out_features);
  
  for (int64_t o = 0; o < out_features; ++o) {
    const float* row = weight + o * in_features;
    QuantParams row_params = compute_symmetric_quant_params(row, in_features);
    params.scales[o] = row_params.scale;
    params.zero_points[o] = row_params.zero_point;
  }
  
  return params;
}

// ============================================================================
// Quantize float tensor to INT8
// ============================================================================
inline void quantize_tensor(
    const float* input, int64_t n,
    const QuantParams& params,
    int8_t* output) {
  
  float inv_scale = 1.0f / params.scale;
  
#ifdef __AVX2__
  __m256 vinv_scale = _mm256_set1_ps(inv_scale);
  __m256 vmin = _mm256_set1_ps(-127.0f);
  __m256 vmax = _mm256_set1_ps(127.0f);
  
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(input + i);
    v = _mm256_mul_ps(v, vinv_scale);
    v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    v = _mm256_max_ps(v, vmin);
    v = _mm256_min_ps(v, vmax);
    
    // Convert to int32
    __m256i vi = _mm256_cvtps_epi32(v);
    
    // Pack to int8 (two-step: int32 -> int16 -> int8)
    __m128i lo = _mm256_castsi256_si128(vi);
    __m128i hi = _mm256_extracti128_si256(vi, 1);
    __m128i packed16 = _mm_packs_epi32(lo, hi);
    __m128i packed8 = _mm_packs_epi16(packed16, packed16);
    
    // Store lower 8 bytes
    _mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), packed8);
  }
  for (; i < n; ++i) {
    float v = input[i] * inv_scale;
    v = std::round(v);
    v = std::max(-127.0f, std::min(127.0f, v));
    output[i] = static_cast<int8_t>(v);
  }
#else
  for (int64_t i = 0; i < n; ++i) {
    float v = input[i] * inv_scale;
    v = std::round(v);
    v = std::max(-127.0f, std::min(127.0f, v));
    output[i] = static_cast<int8_t>(v);
  }
#endif
}

// Per-channel weight quantization
inline void quantize_weight_per_channel(
    const float* weight, int64_t out_features, int64_t in_features,
    const PerChannelQuantParams& params,
    int8_t* output) {
  
  for (int64_t o = 0; o < out_features; ++o) {
    QuantParams row_params(params.scales[o], params.zero_points[o]);
    quantize_tensor(
        weight + o * in_features, in_features,
        row_params,
        output + o * in_features);
  }
}

// ============================================================================
// Dequantize INT8 tensor to float
// ============================================================================
inline void dequantize_tensor(
    const int8_t* input, int64_t n,
    const QuantParams& params,
    float* output) {
  
  float scale = params.scale;
  
#ifdef __AVX2__
  __m256 vscale = _mm256_set1_ps(scale);
  
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    // Load 8 int8 values
    __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + i));
    // Sign-extend int8 -> int16 -> int32
    __m128i shorts = _mm_cvtepi8_epi16(bytes);
    __m256i ints = _mm256_cvtepi16_epi32(shorts);
    // Convert to float
    __m256 floats = _mm256_cvtepi32_ps(ints);
    // Scale
    floats = _mm256_mul_ps(floats, vscale);
    _mm256_storeu_ps(output + i, floats);
  }
  for (; i < n; ++i) {
    output[i] = static_cast<float>(input[i]) * scale;
  }
#else
  for (int64_t i = 0; i < n; ++i) {
    output[i] = static_cast<float>(input[i]) * scale;
  }
#endif
}

// ============================================================================
// Quantized matrix multiplication: C = A @ B^T
// A: (M, K) float, B: (N, K) int8 with per-channel quant, C: (M, N) float
// Mixed precision: FP32 activations, INT8 weights
// ============================================================================
inline void quantized_matmul_mixed(
    const float* A, int64_t M, int64_t K,
    const int8_t* B_quant, int64_t N,
    const PerChannelQuantParams& B_params,
    float* C) {
  
  for (int64_t m = 0; m < M; ++m) {
    const float* a_row = A + m * K;
    
    for (int64_t n = 0; n < N; ++n) {
      const int8_t* b_row = B_quant + n * K;
      float b_scale = B_params.scales[n];
      
      float acc = 0.0f;
      
#ifdef __AVX2__
      __m256 vacc = _mm256_setzero_ps();
      __m256 vscale = _mm256_set1_ps(b_scale);
      
      int64_t k = 0;
      for (; k + 8 <= K; k += 8) {
        __m256 va = _mm256_loadu_ps(a_row + k);
        
        // Load and dequantize B
        __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + k));
        __m128i shorts = _mm_cvtepi8_epi16(bytes);
        __m256i ints = _mm256_cvtepi16_epi32(shorts);
        __m256 vb = _mm256_cvtepi32_ps(ints);
        vb = _mm256_mul_ps(vb, vscale);
        
        vacc = _mm256_fmadd_ps(va, vb, vacc);
      }
      
      // Horizontal sum
      __m128 lo = _mm256_castps256_ps128(vacc);
      __m128 hi = _mm256_extractf128_ps(vacc, 1);
      lo = _mm_add_ps(lo, hi);
      __m128 shuf = _mm_movehdup_ps(lo);
      __m128 sums = _mm_add_ps(lo, shuf);
      shuf = _mm_movehl_ps(shuf, sums);
      sums = _mm_add_ss(sums, shuf);
      acc = _mm_cvtss_f32(sums);
      
      for (; k < K; ++k) {
        acc += a_row[k] * (static_cast<float>(b_row[k]) * b_scale);
      }
#else
      for (int64_t k = 0; k < K; ++k) {
        acc += a_row[k] * (static_cast<float>(b_row[k]) * b_scale);
      }
#endif
      
      C[m * N + n] = acc;
    }
  }
}

// ============================================================================
// Fully INT8 GEMM with INT32 accumulation (for maximum throughput)
// A: (M, K) int8, B: (N, K) int8, C: (M, N) int32
// Final output needs to be dequantized: float_C = int32_C * scale_A * scale_B
// ============================================================================
inline void quantized_matmul_int8(
    const int8_t* A, int64_t M, int64_t K,
    const int8_t* B, int64_t N,
    int32_t* C) {
  
  for (int64_t m = 0; m < M; ++m) {
    const int8_t* a_row = A + m * K;
    
    for (int64_t n = 0; n < N; ++n) {
      const int8_t* b_row = B + n * K;
      
      int32_t acc = 0;
      
#ifdef __AVX2__
      __m256i vacc = _mm256_setzero_si256();
      
      int64_t k = 0;
      for (; k + 32 <= K; k += 32) {
        // Load 32 int8 values from each matrix
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_row + k));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + k));
        
        // Use maddubs for uint8*int8 (need to handle signs carefully)
        // For signed*signed, we use a different approach
        // Split into low/high 128-bit lanes and use madd
        __m128i a_lo = _mm256_castsi256_si128(va);
        __m128i a_hi = _mm256_extracti128_si256(va, 1);
        __m128i b_lo = _mm256_castsi256_si128(vb);
        __m128i b_hi = _mm256_extracti128_si256(vb, 1);
        
        // Sign-extend to 16-bit and multiply-add
        __m256i a_lo_16 = _mm256_cvtepi8_epi16(a_lo);
        __m256i b_lo_16 = _mm256_cvtepi8_epi16(b_lo);
        __m256i prod_lo = _mm256_madd_epi16(a_lo_16, b_lo_16);
        
        __m256i a_hi_16 = _mm256_cvtepi8_epi16(a_hi);
        __m256i b_hi_16 = _mm256_cvtepi8_epi16(b_hi);
        __m256i prod_hi = _mm256_madd_epi16(a_hi_16, b_hi_16);
        
        vacc = _mm256_add_epi32(vacc, prod_lo);
        vacc = _mm256_add_epi32(vacc, prod_hi);
      }
      
      // Horizontal sum of vacc
      __m128i lo = _mm256_castsi256_si128(vacc);
      __m128i hi = _mm256_extracti128_si256(vacc, 1);
      lo = _mm_add_epi32(lo, hi);
      lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
      lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
      acc = _mm_cvtsi128_si32(lo);
      
      for (; k < K; ++k) {
        acc += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b_row[k]);
      }
#else
      for (int64_t k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b_row[k]);
      }
#endif
      
      C[m * N + n] = acc;
    }
  }
}

// Dequantize INT32 result to FP32
inline void dequantize_matmul_result(
    const int32_t* C_int32, int64_t M, int64_t N,
    float scale_A, float scale_B,
    float* C_float) {
  
  float combined_scale = scale_A * scale_B;
  int64_t total = M * N;
  
#ifdef __AVX2__
  __m256 vscale = _mm256_set1_ps(combined_scale);
  int64_t i = 0;
  for (; i + 8 <= total; i += 8) {
    __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C_int32 + i));
    __m256 vf = _mm256_cvtepi32_ps(vi);
    vf = _mm256_mul_ps(vf, vscale);
    _mm256_storeu_ps(C_float + i, vf);
  }
  for (; i < total; ++i) {
    C_float[i] = static_cast<float>(C_int32[i]) * combined_scale;
  }
#else
  for (int64_t i = 0; i < total; ++i) {
    C_float[i] = static_cast<float>(C_int32[i]) * combined_scale;
  }
#endif
}

// ============================================================================
// Quantized Linear layer: y = x @ W^T (W is quantized)
// ============================================================================
struct QuantizedLinear {
  int64_t in_features;
  int64_t out_features;
  std::vector<int8_t> weight_quant;
  PerChannelQuantParams quant_params;
  
  // Quantize from float weights
  void quantize_from_float(const float* weight, int64_t out_feat, int64_t in_feat) {
    out_features = out_feat;
    in_features = in_feat;
    weight_quant.resize(out_features * in_features);
    quant_params = compute_per_channel_quant_params(weight, out_features, in_features);
    quantize_weight_per_channel(weight, out_features, in_features, quant_params, weight_quant.data());
  }
  
  // Forward pass
  void forward(const float* input, int64_t batch_seq, float* output) const {
    quantized_matmul_mixed(
        input, batch_seq, in_features,
        weight_quant.data(), out_features,
        quant_params,
        output);
  }
};

// ============================================================================
// Dynamic quantization helper: quantize activations on-the-fly
// ============================================================================
inline QuantParams dynamic_quantize(
    const float* input, int64_t n,
    int8_t* output) {
  
  QuantParams params = compute_symmetric_quant_params(input, n);
  quantize_tensor(input, n, params, output);
  return params;
}

}  // namespace quant

#endif  // TENSOR_OP_QUANT_HPP

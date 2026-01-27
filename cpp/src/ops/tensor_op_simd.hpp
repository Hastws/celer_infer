// tensor_op_simd.hpp - SIMD (AVX2) optimized tensor operations
// ============================================================================
// SIMD-optimized versions of core operations for improved inference speed.
// Uses AVX2 intrinsics (8-wide float operations).
// ============================================================================

#ifndef TENSOR_OP_SIMD_HPP
#define TENSOR_OP_SIMD_HPP

#include <immintrin.h>  // AVX2 intrinsics

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace simd {

// ============================================================================
// SIMD helper functions
// ============================================================================

// Horizontal sum of 8 floats in __m256
inline float hsum_avx(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

// Fast exp approximation for softmax (polynomial approximation)
// More accurate than naive, sufficient for inference
inline __m256 exp_approx_avx(__m256 x) {
  // Clamp x to avoid overflow
  x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
  x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

  // exp(x) = 2^(x/ln(2)) = 2^(k+f) where k is int, f is fraction
  __m256 log2e = _mm256_set1_ps(1.442695041f);  // 1/ln(2)
  __m256 t = _mm256_mul_ps(x, log2e);

  // Round to nearest integer
  __m256 k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 f = _mm256_sub_ps(t, k);

  // 2^f approximation using polynomial (Horner form)
  // p(f) = c0 + f*(c1 + f*(c2 + f*c3))
  __m256 c0 = _mm256_set1_ps(1.0f);
  __m256 c1 = _mm256_set1_ps(0.693147181f);   // ln(2)
  __m256 c2 = _mm256_set1_ps(0.240226507f);   // ln(2)^2/2
  __m256 c3 = _mm256_set1_ps(0.0558206908f);  // ln(2)^3/6
  __m256 c4 = _mm256_set1_ps(0.00898502902f); // ln(2)^4/24

  __m256 p = _mm256_fmadd_ps(c4, f, c3);
  p = _mm256_fmadd_ps(p, f, c2);
  p = _mm256_fmadd_ps(p, f, c1);
  p = _mm256_fmadd_ps(p, f, c0);

  // 2^k using integer manipulation
  __m256i ki = _mm256_cvtps_epi32(k);
  ki = _mm256_add_epi32(ki, _mm256_set1_epi32(127));
  ki = _mm256_slli_epi32(ki, 23);
  __m256 pow2k = _mm256_castsi256_ps(ki);

  return _mm256_mul_ps(p, pow2k);
}

// Sigmoid approximation
inline __m256 sigmoid_avx(__m256 x) {
  __m256 one = _mm256_set1_ps(1.0f);
  __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
  __m256 exp_neg_x = exp_approx_avx(neg_x);
  return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
}

// ============================================================================
// Element-wise operations (SIMD optimized)
// ============================================================================

inline void silu(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);

  int64_t i = 0;
  // Process 8 floats at a time with AVX2
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(a + i);
    __m256 s = sigmoid_avx(v);
    __m256 result = _mm256_mul_ps(v, s);
    _mm256_storeu_ps(out + i, result);
  }
  // Scalar tail
  for (; i < n; ++i) {
    const float v = a[i];
    out[i] = v / (1.0f + std::exp(-v));
  }
}

inline void swiglu(const float* gate, int64_t n, const float* up, int64_t nu,
                   float* out, int64_t no) {
  assert(n == nu && n == no);

  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 g = _mm256_loadu_ps(gate + i);
    __m256 u = _mm256_loadu_ps(up + i);
    __m256 sg = sigmoid_avx(g);
    __m256 silu_g = _mm256_mul_ps(g, sg);
    __m256 result = _mm256_mul_ps(silu_g, u);
    _mm256_storeu_ps(out + i, result);
  }
  for (; i < n; ++i) {
    const float g = gate[i];
    const float sig = 1.0f / (1.0f + std::exp(-g));
    out[i] = (g * sig) * up[i];
  }
}

// Element-wise multiply
inline void mul(const float* a, int64_t na, const float* b, int64_t nb,
                float* out, int64_t no) {
  assert(na == nb && na == no);
  int64_t i = 0;
  for (; i + 8 <= na; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
  }
  for (; i < na; ++i) out[i] = a[i] * b[i];
}

// Element-wise add
inline void add(const float* a, int64_t na, const float* b, int64_t nb,
                float* out, int64_t no) {
  assert(na == nb && na == no);
  int64_t i = 0;
  for (; i + 8 <= na; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
  }
  for (; i < na; ++i) out[i] = a[i] + b[i];
}

// ============================================================================
// Softmax (SIMD optimized)
// ============================================================================

inline void softmax_lastdim(const float* in, int64_t outer, int64_t inner,
                            float* out) {
  for (int64_t i = 0; i < outer; ++i) {
    const float* row = in + i * inner;
    float* orow = out + i * inner;

    // Find max with SIMD
    __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    int64_t j = 0;
    for (; j + 8 <= inner; j += 8) {
      __m256 v = _mm256_loadu_ps(row + j);
      vmax = _mm256_max_ps(vmax, v);
    }
    float maxval = hsum_avx(_mm256_max_ps(
        _mm256_max_ps(_mm256_permute_ps(vmax, 0x4E),  // swap 128-bit halves
                      vmax),
        _mm256_permute_ps(
            _mm256_max_ps(_mm256_permute_ps(vmax, 0x4E), vmax), 0xB1)));
    // Re-reduce properly
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vmax);
    maxval = tmp[0];
    for (int k = 1; k < 8; ++k)
      if (tmp[k] > maxval) maxval = tmp[k];
    // Scalar tail for max
    for (; j < inner; ++j)
      if (row[j] > maxval) maxval = row[j];

    // Compute exp and sum with SIMD
    __m256 vmaxval = _mm256_set1_ps(maxval);
    __m256 vsum = _mm256_setzero_ps();
    j = 0;
    for (; j + 8 <= inner; j += 8) {
      __m256 v = _mm256_loadu_ps(row + j);
      __m256 e = exp_approx_avx(_mm256_sub_ps(v, vmaxval));
      _mm256_storeu_ps(orow + j, e);
      vsum = _mm256_add_ps(vsum, e);
    }
    float sum = hsum_avx(vsum);
    // Scalar tail
    for (; j < inner; ++j) {
      float e = std::exp(row[j] - maxval);
      orow[j] = e;
      sum += e;
    }

    // Normalize with SIMD
    float inv_sum = 1.0f / sum;
    __m256 vinv = _mm256_set1_ps(inv_sum);
    j = 0;
    for (; j + 8 <= inner; j += 8) {
      __m256 v = _mm256_loadu_ps(orow + j);
      _mm256_storeu_ps(orow + j, _mm256_mul_ps(v, vinv));
    }
    for (; j < inner; ++j) orow[j] *= inv_sum;
  }
}

inline void softmax(const float* x, int64_t x2, int64_t x1, int64_t x0,
                    float* out, int64_t o2, int64_t o1, int64_t o0) {
  assert(x2 == o2 && x1 == o1 && x0 == o0);
  softmax_lastdim(x, x2 * x1, x0, out);
}

// ============================================================================
// RMSNorm (SIMD optimized)
// ============================================================================

inline void rms_norm_lastdim(const float* x, int64_t outer, int64_t a0,
                             const float* weight, int64_t w0, float eps,
                             float* out) {
  assert(a0 == w0);

  for (int64_t i = 0; i < outer; ++i) {
    const float* row = x + i * a0;
    float* orow = out + i * a0;

    // Compute sum of squares with SIMD
    __m256 vss = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 8 <= a0; j += 8) {
      __m256 v = _mm256_loadu_ps(row + j);
      vss = _mm256_fmadd_ps(v, v, vss);
    }
    float ms = hsum_avx(vss);
    // Scalar tail
    for (; j < a0; ++j) ms += row[j] * row[j];

    ms /= static_cast<float>(a0);
    float inv = 1.0f / std::sqrt(ms + eps);
    __m256 vinv = _mm256_set1_ps(inv);

    // Apply normalization with SIMD
    j = 0;
    for (; j + 8 <= a0; j += 8) {
      __m256 vx = _mm256_loadu_ps(row + j);
      __m256 vw = _mm256_loadu_ps(weight + j);
      __m256 result = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw);
      _mm256_storeu_ps(orow + j, result);
    }
    for (; j < a0; ++j) orow[j] = (row[j] * inv) * weight[j];
  }
}

inline void rms_norm(const float* x, int64_t x2, int64_t x1, int64_t x0,
                     const float* weight, int64_t w0, float eps, float* out,
                     int64_t o2, int64_t o1, int64_t o0) {
  assert(x2 == o2 && x1 == o1 && x0 == o0);
  rms_norm_lastdim(x, x2 * x1, x0, weight, w0, eps, out);
}

// ============================================================================
// Matrix Multiplication (SIMD optimized with tiling)
// ============================================================================

// Dot product of two vectors (SIMD)
inline float dot_avx(const float* a, const float* b, int64_t n) {
  __m256 vsum = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    vsum = _mm256_fmadd_ps(va, vb, vsum);
  }
  float sum = hsum_avx(vsum);
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

// Out = A(m,k) @ B(k,n)
// SIMD optimized with loop reordering for better cache utilization
inline void matmul_nn(const float* a, int64_t a1, int64_t a0, const float* b,
                      int64_t b1, int64_t b0, float* out, int64_t o1,
                      int64_t o0) {
  assert(a0 == b1);
  assert(o1 == a1 && o0 == b0);

  const int64_t m = a1, k = a0, n = b0;

  // Zero output
  std::memset(out, 0, m * n * sizeof(float));

  // Block sizes for cache optimization
  constexpr int64_t BM = 32;  // Block size for m
  constexpr int64_t BN = 32;  // Block size for n
  constexpr int64_t BK = 64;  // Block size for k

  for (int64_t bi = 0; bi < m; bi += BM) {
    const int64_t bm_end = std::min(bi + BM, m);
    for (int64_t bj = 0; bj < n; bj += BN) {
      const int64_t bn_end = std::min(bj + BN, n);
      for (int64_t bk = 0; bk < k; bk += BK) {
        const int64_t bk_end = std::min(bk + BK, k);

        // Compute block
        for (int64_t i = bi; i < bm_end; ++i) {
          const float* ai = a + i * k + bk;
          float* oi = out + i * n + bj;
          const int64_t klen = bk_end - bk;

          for (int64_t j = bj; j < bn_end; ++j) {
            // Gather column of B (non-contiguous access)
            // Use SIMD dot product
            __m256 vsum = _mm256_setzero_ps();
            int64_t t = 0;
            for (; t + 8 <= klen; t += 8) {
              __m256 va = _mm256_loadu_ps(ai + t);
              // Gather B column elements
              __m256 vb = _mm256_set_ps(b[(bk + t + 7) * n + j],
                                        b[(bk + t + 6) * n + j],
                                        b[(bk + t + 5) * n + j],
                                        b[(bk + t + 4) * n + j],
                                        b[(bk + t + 3) * n + j],
                                        b[(bk + t + 2) * n + j],
                                        b[(bk + t + 1) * n + j],
                                        b[(bk + t + 0) * n + j]);
              vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            float acc = hsum_avx(vsum);
            for (; t < klen; ++t) acc += ai[t] * b[(bk + t) * n + j];
            oi[j - bj] += acc;
          }
        }
      }
    }
  }
}

// Out = A(m,k) @ B(n,k)^T  - B is transposed (n,k)
// This is cache-friendly as B rows are contiguous
inline void matmul_nt(const float* a, int64_t a1, int64_t a0, const float* b,
                      int64_t b1, int64_t b0, float* out, int64_t o1,
                      int64_t o0) {
  assert(a0 == b0);
  assert(o1 == a1 && o0 == b1);

  const int64_t m = a1, k = a0, n = b1;

  for (int64_t i = 0; i < m; ++i) {
    const float* ai = a + i * k;
    float* oi = out + i * n;
    for (int64_t j = 0; j < n; ++j) {
      const float* bj = b + j * k;
      // SIMD dot product (both are contiguous!)
      oi[j] = dot_avx(ai, bj, k);
    }
  }
}

// 3D batched matmul
inline void matmul_nn(const float* a, int64_t a2, int64_t a1, int64_t a0,
                      const float* b, int64_t b2, int64_t b1, int64_t b0,
                      float* out, int64_t o2, int64_t o1, int64_t o0) {
  assert(a2 == b2);
  assert(a0 == b1);
  assert(o2 == a2 && o1 == a1 && o0 == b0);

  const int64_t batch = a2;
  const int64_t a_stride = a1 * a0;
  const int64_t b_stride = b1 * b0;
  const int64_t o_stride = o1 * o0;

  for (int64_t bi = 0; bi < batch; ++bi) {
    matmul_nn(a + bi * a_stride, a1, a0, b + bi * b_stride, b1, b0,
              out + bi * o_stride, o1, o0);
  }
}

inline void matmul_nt(const float* a, int64_t a2, int64_t a1, int64_t a0,
                      const float* b, int64_t b2, int64_t b1, int64_t b0,
                      float* out, int64_t o2, int64_t o1, int64_t o0) {
  assert(a2 == b2);
  assert(a0 == b0);
  assert(o2 == a2 && o1 == a1 && o0 == b1);

  const int64_t batch = a2;
  const int64_t a_stride = a1 * a0;
  const int64_t b_stride = b1 * b0;
  const int64_t o_stride = o1 * b1;

  for (int64_t bi = 0; bi < batch; ++bi) {
    matmul_nt(a + bi * a_stride, a1, a0, b + bi * b_stride, b1, b0,
              out + bi * o_stride, o1, b1);
  }
}

// 4D batched matmul
inline void matmul_nn(const float* a, int64_t a3, int64_t a2, int64_t a1,
                      int64_t a0, const float* b, int64_t b3, int64_t b2,
                      int64_t b1, int64_t b0, float* out, int64_t o3,
                      int64_t o2, int64_t o1, int64_t o0) {
  assert(a3 == b3 && a2 == b2);
  const int64_t outer = a3 * a2;
  matmul_nn(a, outer, a1, a0, b, outer, b1, b0, out, outer, o1, o0);
}

inline void matmul_nt(const float* a, int64_t a3, int64_t a2, int64_t a1,
                      int64_t a0, const float* b, int64_t b3, int64_t b2,
                      int64_t b1, int64_t b0, float* out, int64_t o3,
                      int64_t o2, int64_t o1, int64_t o0) {
  assert(a3 == b3 && a2 == b2);
  const int64_t outer = a3 * a2;
  matmul_nt(a, outer, a1, a0, b, outer, b1, b0, out, outer, o1, b1);
}

// ============================================================================
// Attention-specific matmuls (optimized for Q@K^T and P@V patterns)
// ============================================================================

// Q @ K^T : Q(B,H,S,D) @ K(B,H,S,D)^T -> (B,H,S,S)
inline void attn_qk_matmul(const float* q, int64_t qb, int64_t qh, int64_t qs,
                           int64_t qd, const float* k, int64_t kb, int64_t kh,
                           int64_t ks, int64_t kd, float* out, int64_t ob,
                           int64_t oh, int64_t os1, int64_t os2) {
  assert(qb == kb && qh == kh && qd == kd && qs == ks);
  assert(ob == qb && oh == qh && os1 == qs && os2 == ks);

  const int64_t batch = qb * qh;
  const int64_t q_stride = qs * qd;
  const int64_t k_stride = ks * kd;
  const int64_t o_stride = os1 * os2;

  for (int64_t bi = 0; bi < batch; ++bi) {
    const float* qi = q + bi * q_stride;
    const float* ki = k + bi * k_stride;
    float* oi = out + bi * o_stride;

    for (int64_t si = 0; si < qs; ++si) {
      const float* qrow = qi + si * qd;
      for (int64_t sj = 0; sj < ks; ++sj) {
        const float* krow = ki + sj * kd;
        oi[si * os2 + sj] = dot_avx(qrow, krow, qd);
      }
    }
  }
}

// P @ V : P(B,H,S,S) @ V(B,H,S,D) -> (B,H,S,D)
inline void attn_pv_matmul(const float* p, int64_t pb, int64_t ph, int64_t ps1,
                           int64_t ps2, const float* v, int64_t vb, int64_t vh,
                           int64_t vs, int64_t vd, float* out, int64_t ob,
                           int64_t oh, int64_t os, int64_t od) {
  assert(pb == vb && ph == vh && ps2 == vs);
  assert(ob == pb && oh == ph && os == ps1 && od == vd);

  const int64_t batch = pb * ph;
  const int64_t p_stride = ps1 * ps2;
  const int64_t v_stride = vs * vd;
  const int64_t o_stride = os * od;

  for (int64_t bi = 0; bi < batch; ++bi) {
    const float* pi = p + bi * p_stride;
    const float* vi = v + bi * v_stride;
    float* oi = out + bi * o_stride;

    // For each output row
    for (int64_t si = 0; si < ps1; ++si) {
      const float* prow = pi + si * ps2;
      float* orow = oi + si * od;

      // Initialize output row to zero
      std::memset(orow, 0, od * sizeof(float));

      // Weighted sum: orow += prow[sj] * V[sj,:]
      for (int64_t sj = 0; sj < ps2; ++sj) {
        const float pval = prow[sj];
        const float* vrow = vi + sj * vd;

        __m256 vp = _mm256_set1_ps(pval);
        int64_t d = 0;
        for (; d + 8 <= od; d += 8) {
          __m256 vo = _mm256_loadu_ps(orow + d);
          __m256 vv = _mm256_loadu_ps(vrow + d);
          vo = _mm256_fmadd_ps(vp, vv, vo);
          _mm256_storeu_ps(orow + d, vo);
        }
        for (; d < od; ++d) orow[d] += pval * vrow[d];
      }
    }
  }
}

// ============================================================================
// Copy operations (simple memcpy wrappers)
// ============================================================================

inline void copy(const float* src, int64_t n, float* dst, int64_t dn) {
  assert(n == dn);
  std::memcpy(dst, src, n * sizeof(float));
}

// ============================================================================
// Scale operations
// ============================================================================

inline void scale(float* data, int64_t n, float s) {
  __m256 vs = _mm256_set1_ps(s);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(data + i);
    _mm256_storeu_ps(data + i, _mm256_mul_ps(v, vs));
  }
  for (; i < n; ++i) data[i] *= s;
}

// ============================================================================
// Causal mask for attention
// ============================================================================

inline void apply_causal_mask(float* attn, int64_t batch, int64_t heads,
                              int64_t seq_q, int64_t seq_k) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const int64_t outer = batch * heads;
  const int64_t stride = seq_q * seq_k;

  for (int64_t b = 0; b < outer; ++b) {
    float* mat = attn + b * stride;
    for (int64_t i = 0; i < seq_q; ++i) {
      for (int64_t j = i + 1; j < seq_k; ++j) {
        mat[i * seq_k + j] = neg_inf;
      }
    }
  }
}

}  // namespace simd

#endif  // TENSOR_OP_SIMD_HPP

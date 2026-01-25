#ifndef CELER_INFER_INCLUDE_CELER_INFER_TENSOR_HPP
#define CELER_INFER_INCLUDE_CELER_INFER_TENSOR_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace celer_infer {
// ============================================================================
// Shape convention = "negative indexing"
//
// a0 == size(-1) : contiguous dim (stride=1)
// a1 == size(-2)
// a2 == size(-3)
// a3 == size(-4)
// a4 == size(-5)
//
// Row-major contiguous layout (C-style):
// 2D offset: idx = i1 * a0 + i0
// 3D offset: idx = (i2 * a1 + i1) * a0 + i0
// 5D offset: idx = ((((i4 * a3 + i3) * a2 + i2) * a1 + i1) * a0 + i0)
// ============================================================================

inline float neg_inf_f32() { return -std::numeric_limits<float>::infinity(); }

inline int64_t numel_1d(int64_t a0) { return a0; }
inline int64_t numel_2d(int64_t a1, int64_t a0) { return a1 * a0; }
inline int64_t numel_3d(int64_t a2, int64_t a1, int64_t a0) {
  return a2 * a1 * a0;
}
inline int64_t numel_4d(int64_t a3, int64_t a2, int64_t a1, int64_t a0) {
  return a3 * a2 * a1 * a0;
}

inline int64_t numel_5d(int64_t a4, int64_t a3, int64_t a2, int64_t a1,
                        int64_t a0) {
  return a4 * a3 * a2 * a1 * a0;
}

inline int64_t off_2d(int64_t i1, int64_t i0, int64_t a0) {
  return i1 * a0 + i0;
}

inline int64_t off_3d(int64_t i2, int64_t i1, int64_t i0, int64_t a1,
                      int64_t a0) {
  return (i2 * a1 + i1) * a0 + i0;
}

inline int64_t off_4d(int64_t i3, int64_t i2, int64_t i1, int64_t i0,
                      int64_t a2, int64_t a1, int64_t a0) {
  return ((i3 * a2 + i2) * a1 + i1) * a0 + i0;
}

inline int64_t off_5d(int64_t i4, int64_t i3, int64_t i2, int64_t i1,
                      int64_t i0, int64_t a3, int64_t a2, int64_t a1,
                      int64_t a0) {
  return (((i4 * a3 + i3) * a2 + i2) * a1 + i1) * a0 + i0;
}

// ============================================================================
// Fill / Copy
// ============================================================================

inline void fill(float* out, int64_t o0, float v) {
  for (int64_t i = 0; i < o0; ++i) out[i] = v;
}

inline void fill(float* out, int64_t o1, int64_t o0, float v) {
  fill(out, o1 * o0, v);
}
inline void fill(float* out, int64_t o2, int64_t o1, int64_t o0, float v) {
  fill(out, o2 * o1 * o0, v);
}

inline void fill(float* out, int64_t o3, int64_t o2, int64_t o1, int64_t o0,
                 float v) {
  fill(out, o3 * o2 * o1 * o0, v);
}

inline void fill(float* out, int64_t o4, int64_t o3, int64_t o2, int64_t o1,
                 int64_t o0, float v) {
  fill(out, o4 * o3 * o2 * o1 * o0, v);
}

inline void copy(const float* a, int64_t a0, float* out, int64_t o0) {
  assert(a0 == o0);
  std::memcpy(out, a, static_cast<size_t>(a0) * sizeof(float));
}

inline void copy(const float* a, int64_t a1, int64_t a0, float* out, int64_t o1,
                 int64_t o0) {
  assert(a1 == o1 && a0 == o0);
  copy(a, a1 * a0, out, o1 * o0);
}

inline void copy(const float* a, int64_t a2, int64_t a1, int64_t a0, float* out,
                 int64_t o2, int64_t o1, int64_t o0) {
  assert(a2 == o2 && a1 == o1 && a0 == o0);
  copy(a, a2 * a1 * a0, out, o2 * o1 * o0);
}

inline void copy(const float* a, int64_t a3, int64_t a2, int64_t a1, int64_t a0,
                 float* out, int64_t o3, int64_t o2, int64_t o1, int64_t o0) {
  assert(a3 == o3 && a2 == o2 && a1 == o1 && a0 == o0);
  copy(a, a3 * a2 * a1 * a0, out, o3 * o2 * o1 * o0);
}

inline void copy(const float* a, int64_t a4, int64_t a3, int64_t a2, int64_t a1,
                 int64_t a0, float* out, int64_t o4, int64_t o3, int64_t o2,
                 int64_t o1, int64_t o0) {
  assert(a4 == o4 && a3 == o3 && a2 == o2 && a1 == o1 && a0 == o0);
  copy(a, a4 * a3 * a2 * a1 * a0, out, o4 * o3 * o2 * o1 * o0);
}

// ============================================================================
// Elementwise (same-shape) : add/sub/mul/div, in-place variants
// ============================================================================

inline void add(const float* a, int64_t n, const float* b, int64_t nb,
                float* out, int64_t no) {
  assert(n == nb && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

inline void sub(const float* a, int64_t n, const float* b, int64_t nb,
                float* out, int64_t no) {
  assert(n == nb && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

inline void mul(const float* a, int64_t n, const float* b, int64_t nb,
                float* out, int64_t no) {
  assert(n == nb && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

inline void div(const float* a, int64_t n, const float* b, int64_t nb,
                float* out, int64_t no) {
  assert(n == nb && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
}

inline void add_inplace(float* a, int64_t n, const float* b, int64_t nb) {
  assert(n == nb);
  for (int64_t i = 0; i < n; ++i) a[i] += b[i];
}

inline void mul_inplace(float* a, int64_t n, const float* b, int64_t nb) {
  assert(n == nb);
  for (int64_t i = 0; i < n; ++i) a[i] *= b[i];
}

inline void add(const float* a, int64_t a1, int64_t a0, const float* b,
                int64_t b1, int64_t b0, float* out, int64_t o1, int64_t o0) {
  assert(a1 == b1 && a0 == b0 && a1 == o1 && a0 == o0);
  add(a, a1 * a0, b, b1 * b0, out, o1 * o0);
}

inline void add(const float* a, int64_t a2, int64_t a1, int64_t a0,
                const float* b, int64_t b2, int64_t b1, int64_t b0, float* out,
                int64_t o2, int64_t o1, int64_t o0) {
  assert(a2 == b2 && a1 == b1 && a0 == b0 && a2 == o2 && a1 == o1 && a0 == o0);
  add(a, a2 * a1 * a0, b, b2 * b1 * b0, out, o2 * o1 * o0);
}

inline void add(const float* a, int64_t a3, int64_t a2, int64_t a1, int64_t a0,
                const float* b, int64_t b3, int64_t b2, int64_t b1, int64_t b0,
                float* out, int64_t o3, int64_t o2, int64_t o1, int64_t o0) {
  assert(a3 == b3 && a2 == b2 && a1 == b1 && a0 == b0);
  assert(a3 == o3 && a2 == o2 && a1 == o1 && a0 == o0);
  add(a, a3 * a2 * a1 * a0, b, b3 * b2 * b1 * b0, out, o3 * o2 * o1 * o0);
}

inline void add(const float* a, int64_t a4, int64_t a3, int64_t a2, int64_t a1,
                int64_t a0, const float* b, int64_t b4, int64_t b3, int64_t b2,
                int64_t b1, int64_t b0, float* out, int64_t o4, int64_t o3,
                int64_t o2, int64_t o1, int64_t o0) {
  assert(a4 == b4 && a3 == b3 && a2 == b2 && a1 == b1 && a0 == b0);
  assert(a4 == o4 && a3 == o3 && a2 == o2 && a1 == o1 && a0 == o0);
  add(a, a4 * a3 * a2 * a1 * a0, b, b4 * b3 * b2 * b1 * b0, out,
      o4 * o3 * o2 * o1 * o0);
}

// ============================================================================
// Scalar ops
// ============================================================================

inline void scale_inplace(float* x, int64_t n, float s) {
  for (int64_t i = 0; i < n; ++i) x[i] *= s;
}

inline void add_scalar(const float* a, int64_t n, float s, float* out,
                       int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] + s;
}

inline void mul_scalar(const float* a, int64_t n, float s, float* out,
                       int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] * s;
}

inline void clamp(const float* a, int64_t n, float lo, float hi, float* out,
                  int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = std::min(std::max(a[i], lo), hi);
}

// ============================================================================
// Unary ops (float)
// ============================================================================

inline float sigmoid_f32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline void exp(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = std::exp(a[i]);
}

inline void log(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = std::log(a[i]);
}

inline void sqrt(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = std::sqrt(a[i]);
}

inline void rsqrt(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = 1.0f / std::sqrt(a[i]);
}

inline void tanh(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = std::tanh(a[i]);
}

inline void sigmoid(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = sigmoid_f32(a[i]);
}

inline void relu(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
}

inline void silu(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) {
    const float v = a[i];
    out[i] = v * sigmoid_f32(v);
  }
}

// GELU tanh approximation
inline void gelu(const float* a, int64_t n, float* out, int64_t no) {
  assert(n == no);
  const float k = std::sqrt(2.0f / 3.14159265358979323846f);
  for (int64_t i = 0; i < n; ++i) {
    const float x = a[i];
    const float x3 = x * x * x;
    out[i] = 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x3)));
  }
}

// SwiGLU: out = silu(gate) * up
inline void swiglu(const float* gate, int64_t n, const float* up, int64_t nu,
                   float* out, int64_t no) {
  assert(n == nu && n == no);
  for (int64_t i = 0; i < n; ++i) {
    const float g = gate[i];
    out[i] = (g * sigmoid_f32(g)) * up[i];
  }
}

// ============================================================================
// Last-dim vector broadcast (typical for bias/weight on hidden dim)
// x[..., a0] op w[a0]
// ============================================================================

inline void add_lastdim_vec(const float* x, int64_t outer, int64_t a0,
                            const float* w, int64_t w0, float* out) {
  assert(a0 == w0);
  for (int64_t i = 0; i < outer; ++i) {
    const float* xi = x + i * a0;
    float* yi = out + i * a0;
    for (int64_t j = 0; j < a0; ++j) yi[j] = xi[j] + w[j];
  }
}

inline void mul_lastdim_vec(const float* x, int64_t outer, int64_t a0,
                            const float* w, int64_t w0, float* out) {
  assert(a0 == w0);
  for (int64_t i = 0; i < outer; ++i) {
    const float* xi = x + i * a0;
    float* yi = out + i * a0;
    for (int64_t j = 0; j < a0; ++j) yi[j] = xi[j] * w[j];
  }
}

// Overloads (2D~5D)
inline void add_lastdim_vec(const float* x, int64_t x1, int64_t x0,
                            const float* w, int64_t w0, float* out, int64_t o1,
                            int64_t o0) {
  assert(x1 == o1 && x0 == o0);
  add_lastdim_vec(x, x1, x0, w, w0, out);
}

inline void add_lastdim_vec(const float* x, int64_t x2, int64_t x1, int64_t x0,
                            const float* w, int64_t w0, float* out, int64_t o2,
                            int64_t o1, int64_t o0) {
  assert(x2 == o2 && x1 == o1 && x0 == o0);
  add_lastdim_vec(x, x2 * x1, x0, w, w0, out);
}

inline void add_lastdim_vec(const float* x, int64_t x3, int64_t x2, int64_t x1,
                            int64_t x0, const float* w, int64_t w0, float* out,
                            int64_t o3, int64_t o2, int64_t o1, int64_t o0) {
  assert(x3 == o3 && x2 == o2 && x1 == o1 && x0 == o0);
  add_lastdim_vec(x, x3 * x2 * x1, x0, w, w0, out);
}

inline void add_lastdim_vec(const float* x, int64_t x4, int64_t x3, int64_t x2,
                            int64_t x1, int64_t x0, const float* w, int64_t w0,
                            float* out, int64_t o4, int64_t o3, int64_t o2,
                            int64_t o1, int64_t o0) {
  assert(x4 == o4 && x3 == o3 && x2 == o2 && x1 == o1 && x0 == o0);
  add_lastdim_vec(x, x4 * x3 * x2 * x1, x0, w, w0, out);
}

// ============================================================================
// Reductions along last dim (a0): sum/mean/max
// Input dims: (..., a0) -> output dims: (...) (flattened as outer)
// ============================================================================

inline void sum_lastdim(const float* x, int64_t outer, int64_t a0, float* out,
                        int64_t o0) {
  assert(outer == o0);
  for (int64_t i = 0; i < outer; ++i) {
    const float* row = x + i * a0;
    double acc = 0.0;
    for (int64_t j = 0; j < a0; ++j) acc += static_cast<double>(row[j]);
    out[i] = static_cast<float>(acc);
  }
}

inline void mean_lastdim(const float* x, int64_t outer, int64_t a0, float* out,
                         int64_t o0) {
  sum_lastdim(x, outer, a0, out, o0);
  const float inv = 1.0f / static_cast<float>(a0);
  for (int64_t i = 0; i < o0; ++i) out[i] *= inv;
}

inline void max_lastdim(const float* x, int64_t outer, int64_t a0, float* out,
                        int64_t o0) {
  assert(outer == o0);
  for (int64_t i = 0; i < outer; ++i) {
    const float* row = x + i * a0;
    float mx = row[0];
    for (int64_t j = 1; j < a0; ++j) mx = std::max(mx, row[j]);
    out[i] = mx;
  }
}

// General reduce_sum for 5D along axis in {-5,-4,-3,-2,-1} (slow but “全”)
// Output dims are caller-provided (axis removed).
inline void reduce_sum_5d_axis(const float* x, int64_t a4, int64_t a3,
                               int64_t a2, int64_t a1, int64_t a0,
                               int32_t axis_neg, float* out, int64_t o4,
                               int64_t o3, int64_t o2, int64_t o1, int64_t o0) {
  assert(axis_neg <= -1 && axis_neg >= -5);
  const int axis = axis_neg + 5;  // map -5..-1 -> 0..4
  const int64_t ad[5] = {a4, a3, a2, a1, a0};
  int64_t od[5] = {o4, o3, o2, o1, o0};
  // Validate output dims match input dims with one axis removed.
  for (int i = 0, j = 0; i < 5; ++i) {
    if (i == axis) continue;
    assert(od[j] == ad[i]);
    ++j;
  }
  // Fill out to zero
  fill(out, o4 * o3 * o2 * o1 * o0, 0.0f);

  // Brute-force loop over all input indices, accumulate to output.
  for (int64_t i4 = 0; i4 < a4; ++i4) {
    for (int64_t i3 = 0; i3 < a3; ++i3) {
      for (int64_t i2 = 0; i2 < a2; ++i2) {
        for (int64_t i1 = 0; i1 < a1; ++i1) {
          for (int64_t i0 = 0; i0 < a0; ++i0) {
            const float v = x[off_5d(i4, i3, i2, i1, i0, a3, a2, a1, a0)];

            // Build output indices by skipping axis.
            int64_t oi[4];
            int t = 0;
            const int64_t ii[5] = {i4, i3, i2, i1, i0};
            for (int k = 0; k < 5; ++k) {
              if (k == axis) continue;
              oi[t++] = ii[k];
            }
            // Output is 4D packed as (o4,o3,o2,o1,o0) but only first 4 are
            // meaningful. We interpret output as contiguous with dims exactly
            // as provided.
            const int64_t out_idx =
                (((oi[0] * o3 + oi[1]) * o2 + oi[2]) * o1 + oi[3]) * o0;
            // o0 is 1 for true 4D output
            out[out_idx] += v;
          }
        }
      }
    }
  }
}

// ============================================================================
// Softmax / LogSoftmax along last dim (a0), stable
// ============================================================================

inline void softmax_lastdim(const float* x, int64_t outer, int64_t a0,
                            float* out) {
  for (int64_t i = 0; i < outer; ++i) {
    const float* row = x + i * a0;
    float* orow = out + i * a0;

    float mx = row[0];
    for (int64_t j = 1; j < a0; ++j) mx = std::max(mx, row[j]);

    double sum = 0.0;
    for (int64_t j = 0; j < a0; ++j) {
      const float e = std::exp(row[j] - mx);
      orow[j] = e;
      sum += static_cast<double>(e);
    }
    const float inv = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
    for (int64_t j = 0; j < a0; ++j) orow[j] *= inv;
  }
}

inline void log_softmax_lastdim(const float* x, int64_t outer, int64_t a0,
                                float* out) {
  for (int64_t i = 0; i < outer; ++i) {
    const float* row = x + i * a0;
    float* orow = out + i * a0;

    float mx = row[0];
    for (int64_t j = 1; j < a0; ++j) mx = std::max(mx, row[j]);

    double sum = 0.0;
    for (int64_t j = 0; j < a0; ++j)
      sum += static_cast<double>(std::exp(row[j] - mx));
    const float logsum =
        (sum > 0.0) ? static_cast<float>(std::log(sum)) : neg_inf_f32();
    for (int64_t j = 0; j < a0; ++j) orow[j] = (row[j] - mx) - logsum;
  }
}

// Wrappers (2D~5D)
inline void softmax(const float* x, int64_t x1, int64_t x0, float* out,
                    int64_t o1, int64_t o0) {
  assert(x1 == o1 && x0 == o0);
  softmax_lastdim(x, x1, x0, out);
}

inline void softmax(const float* x, int64_t x2, int64_t x1, int64_t x0,
                    float* out, int64_t o2, int64_t o1, int64_t o0) {
  assert(x2 == o2 && x1 == o1 && x0 == o0);
  softmax_lastdim(x, x2 * x1, x0, out);
}

inline void softmax(const float* x, int64_t x3, int64_t x2, int64_t x1,
                    int64_t x0, float* out, int64_t o3, int64_t o2, int64_t o1,
                    int64_t o0) {
  assert(x3 == o3 && x2 == o2 && x1 == o1 && x0 == o0);
  softmax_lastdim(x, x3 * x2 * x1, x0, out);
}

inline void softmax(const float* x, int64_t x4, int64_t x3, int64_t x2,
                    int64_t x1, int64_t x0, float* out, int64_t o4, int64_t o3,
                    int64_t o2, int64_t o1, int64_t o0) {
  assert(x4 == o4 && x3 == o3 && x2 == o2 && x1 == o1 && x0 == o0);
  softmax_lastdim(x, x4 * x3 * x2 * x1, x0, out);
}

inline void log_softmax(const float* x, int64_t x1, int64_t x0, float* out,
                        int64_t o1, int64_t o0) {
  assert(x1 == o1 && x0 == o0);
  log_softmax_lastdim(x, x1, x0, out);
}

// ============================================================================
// Matmul family
// 2D: A(a1,a0) @ B(b1,b0) -> Out(o1,o0)
// Negative-index mapping:
//   A: (m,k) => a1=m, a0=k
//   B: (k,n) => b1=k, b0=n
//   Out:(m,n)=> o1=m, o0=n
// ============================================================================

inline void matmul_nn(const float* a, int64_t a1, int64_t a0, const float* b,
                      int64_t b1, int64_t b0, float* out, int64_t o1,
                      int64_t o0) {
  assert(a0 == b1);
  assert(o1 == a1 && o0 == b0);

  const int64_t m = a1, k = a0, n = b0;
  for (int64_t i = 0; i < m; ++i) {
    const float* ai = a + i * k;
    float* oi = out + i * n;
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t t = 0; t < k; ++t)
        acc += static_cast<double>(ai[t]) * static_cast<double>(b[t * n + j]);
      oi[j] = static_cast<float>(acc);
    }
  }
}

// Out = A(m,k) @ B(n,k)^T  (B stored as (n,k): b1=n, b0=k)
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
      double acc = 0.0;
      for (int64_t t = 0; t < k; ++t)
        acc += static_cast<double>(ai[t]) * static_cast<double>(bj[t]);
      oi[j] = static_cast<float>(acc);
    }
  }
}

// Batched: treat leading dims as batch, last two dims as matmul dims.
// 3D: A(a2,a1,a0) where (a1,a0) = (m,k), batch=a2
inline void matmul_nn(const float* a, int64_t a2, int64_t a1, int64_t a0,
                      const float* b, int64_t b2, int64_t b1, int64_t b0,
                      float* out, int64_t o2, int64_t o1, int64_t o0) {
  assert(a2 == b2);
  assert(a0 == b1);
  assert(o2 == a2 && o1 == a1 && o0 == b0);

  const int64_t batch = a2;
  const int64_t m = a1, k = a0, n = b0;
  const int64_t a_stride = m * k;
  const int64_t b_stride = k * n;
  const int64_t o_stride = m * n;
  for (int64_t bi = 0; bi < batch; ++bi) {
    matmul_nn(a + bi * a_stride, m, k, b + bi * b_stride, k, n,
              out + bi * o_stride, m, n);
  }
}

// 4D: A(a3,a2,a1,a0), B(b3,b2,b1,b0) -> batch = a3*a2, mat dims = (a1,a0)
inline void matmul_nn(const float* a, int64_t a3, int64_t a2, int64_t a1,
                      int64_t a0, const float* b, int64_t b3, int64_t b2,
                      int64_t b1, int64_t b0, float* out, int64_t o3,
                      int64_t o2, int64_t o1, int64_t o0) {
  assert(a3 == b3 && a2 == b2);
  assert(a0 == b1);
  assert(o3 == a3 && o2 == a2 && o1 == a1 && o0 == b0);

  const int64_t batch = a3 * a2;
  const int64_t m = a1, k = a0, n = b0;
  const int64_t a_stride = m * k;
  const int64_t b_stride = k * n;
  const int64_t o_stride = m * n;
  for (int64_t bi = 0; bi < batch; ++bi) {
    matmul_nn(a + bi * a_stride, m, k, b + bi * b_stride, k, n,
              out + bi * o_stride, m, n);
  }
}

// 5D: batch = a4*a3*a2, mat dims = (a1,a0)
inline void matmul_nn(const float* a, int64_t a4, int64_t a3, int64_t a2,
                      int64_t a1, int64_t a0, const float* b, int64_t b4,
                      int64_t b3, int64_t b2, int64_t b1, int64_t b0,
                      float* out, int64_t o4, int64_t o3, int64_t o2,
                      int64_t o1, int64_t o0) {
  assert(a4 == b4 && a3 == b3 && a2 == b2);
  assert(a0 == b1);
  assert(o4 == a4 && o3 == a3 && o2 == a2 && o1 == a1 && o0 == b0);

  const int64_t batch = a4 * a3 * a2;
  const int64_t m = a1, k = a0, n = b0;
  const int64_t a_stride = m * k;
  const int64_t b_stride = k * n;
  const int64_t o_stride = m * n;
  for (int64_t bi = 0; bi < batch; ++bi) {
    matmul_nn(a + bi * a_stride, m, k, b + bi * b_stride, k, n,
              out + bi * o_stride, m, n);
  }
}

// Linear (no bias): x(..., in) @ w(out, in)^T  where w dims are (out,in):
// w1=out,w0=in This matches PyTorch nn.Linear without bias and weight layout
// [out_features, in_features].
inline void linear_no_bias(const float* x, int64_t x2, int64_t x1, int64_t x0,
                           const float* w, int64_t w1, int64_t w0, float* out,
                           int64_t o2, int64_t o1, int64_t o0) {
  assert(x2 == o2 && x1 == o1);
  assert(w0 == x0);
  assert(o0 == w1);

  const int64_t outer = x2 * x1;
  // out_flat(outer, out_features) = x_flat(outer, in) @ w(out,in)^T
  // which is matmul_nt(x_flat, w)
  matmul_nt(x, outer, x0, w, w1, w0, out, outer, w1);
}

// ============================================================================
// Transpose / Permute
// ============================================================================

// 2D transpose: input (a1,a0) -> output (a0,a1)
inline void transpose_2d(const float* a, int64_t a1, int64_t a0, float* out,
                         int64_t o1, int64_t o0) {
  assert(o1 == a0 && o0 == a1);
  for (int64_t i1 = 0; i1 < a1; ++i1) {
    for (int64_t i0 = 0; i0 < a0; ++i0) {
      out[off_2d(i0, i1, o0)] = a[off_2d(i1, i0, a0)];
    }
  }
}

// Generic permute for 5D (runtime perm[5]):
// perm maps output dim -> input dim, each in [0..4], using (a4,a3,a2,a1,a0).
inline void permute_5d(const float* x, int64_t a4, int64_t a3, int64_t a2,
                       int64_t a1, int64_t a0, const int32_t* perm5, float* out,
                       int64_t o4, int64_t o3, int64_t o2, int64_t o1,
                       int64_t o0) {
  const int64_t in_dims[5] = {a4, a3, a2, a1, a0};
  const int64_t out_dims[5] = {o4, o3, o2, o1, o0};
  for (int i = 0; i < 5; ++i) {
    const int src = perm5[i];
    assert(0 <= src && src < 5);
    assert(out_dims[i] == in_dims[src]);
  }

  const int64_t in_stride[5] = {a3 * a2 * a1 * a0, a2 * a1 * a0, a1 * a0, a0,
                                1};
  const int64_t out_stride[5] = {o3 * o2 * o1 * o0, o2 * o1 * o0, o1 * o0, o0,
                                 1};

  const int64_t total = o4 * o3 * o2 * o1 * o0;
  for (int64_t idx = 0; idx < total; ++idx) {
    int64_t t = idx;
    int64_t oi[5];
    for (int i = 0; i < 5; ++i) {
      oi[i] = t / out_stride[i];
      t %= out_stride[i];
    }
    int64_t in_off = 0;
    for (int i = 0; i < 5; ++i) in_off += oi[i] * in_stride[perm5[i]];
    out[idx] = x[in_off];
  }
}

// ============================================================================
// Slice / Pad / Concat (5D generic)
// ============================================================================

inline void slice_5d(const float* x, int64_t a4, int64_t a3, int64_t a2,
                     int64_t a1, int64_t a0, int64_t s4, int64_t s3, int64_t s2,
                     int64_t s1, int64_t s0, int64_t n4, int64_t n3, int64_t n2,
                     int64_t n1, int64_t n0, float* out, int64_t o4, int64_t o3,
                     int64_t o2, int64_t o1, int64_t o0) {
  assert(o4 == n4 && o3 == n3 && o2 == n2 && o1 == n1 && o0 == n0);
  assert(0 <= s4 && s4 + n4 <= a4);
  assert(0 <= s3 && s3 + n3 <= a3);
  assert(0 <= s2 && s2 + n2 <= a2);
  assert(0 <= s1 && s1 + n1 <= a1);
  assert(0 <= s0 && s0 + n0 <= a0);

  for (int64_t i4 = 0; i4 < n4; ++i4) {
    for (int64_t i3 = 0; i3 < n3; ++i3) {
      for (int64_t i2 = 0; i2 < n2; ++i2) {
        for (int64_t i1 = 0; i1 < n1; ++i1) {
          const float* src = x + off_5d(s4 + i4, s3 + i3, s2 + i2, s1 + i1, s0,
                                        a3, a2, a1, a0);
          float* dst = out + off_5d(i4, i3, i2, i1, 0, o3, o2, o1, o0);
          std::memcpy(dst, src, static_cast<size_t>(n0) * sizeof(float));
        }
      }
    }
  }
}

inline void pad_5d_constant(const float* x, int64_t a4, int64_t a3, int64_t a2,
                            int64_t a1, int64_t a0, int64_t p4l, int64_t p4r,
                            int64_t p3l, int64_t p3r, int64_t p2l, int64_t p2r,
                            int64_t p1l, int64_t p1r, int64_t p0l, int64_t p0r,
                            float val, float* out, int64_t o4, int64_t o3,
                            int64_t o2, int64_t o1, int64_t o0) {
  assert(o4 == a4 + p4l + p4r);
  assert(o3 == a3 + p3l + p3r);
  assert(o2 == a2 + p2l + p2r);
  assert(o1 == a1 + p1l + p1r);
  assert(o0 == a0 + p0l + p0r);

  fill(out, o4 * o3 * o2 * o1 * o0, val);

  for (int64_t i4 = 0; i4 < a4; ++i4) {
    for (int64_t i3 = 0; i3 < a3; ++i3) {
      for (int64_t i2 = 0; i2 < a2; ++i2) {
        for (int64_t i1 = 0; i1 < a1; ++i1) {
          const float* src = x + off_5d(i4, i3, i2, i1, 0, a3, a2, a1, a0);
          float* dst = out + off_5d(i4 + p4l, i3 + p3l, i2 + p2l, i1 + p1l, p0l,
                                    o3, o2, o1, o0);
          std::memcpy(dst, src, static_cast<size_t>(a0) * sizeof(float));
        }
      }
    }
  }
}

inline void concat_5d_axis(const float* a, int64_t a4, int64_t a3, int64_t a2,
                           int64_t a1, int64_t a0, const float* b, int64_t b4,
                           int64_t b3, int64_t b2, int64_t b1, int64_t b0,
                           int32_t axis_neg, float* out, int64_t o4, int64_t o3,
                           int64_t o2, int64_t o1, int64_t o0) {
  assert(axis_neg <= -1 && axis_neg >= -5);
  const int axis = axis_neg + 5;  // -5..-1 -> 0..4

  const int64_t ad[5] = {a4, a3, a2, a1, a0};
  const int64_t bd[5] = {b4, b3, b2, b1, b0};
  const int64_t od[5] = {o4, o3, o2, o1, o0};

  for (int i = 0; i < 5; ++i) {
    if (i == axis) {
      assert(od[i] == ad[i] + bd[i]);
    } else {
      assert(ad[i] == bd[i] && od[i] == ad[i]);
    }
  }

  int64_t pre = 1;
  for (int i = 0; i < axis; ++i) pre *= od[i];
  int64_t post = 1;
  for (int i = axis + 1; i < 5; ++i) post *= od[i];

  const int64_t a_axis = ad[axis];
  const int64_t b_axis = bd[axis];
  const int64_t o_axis = od[axis];

  for (int64_t p = 0; p < pre; ++p) {
    const int64_t a_off = p * (a_axis * post);
    const int64_t o_off_a = p * (o_axis * post);
    std::memcpy(out + o_off_a, a + a_off,
                static_cast<size_t>(a_axis * post) * sizeof(float));

    const int64_t b_off = p * (b_axis * post);
    const int64_t o_off_b = o_off_a + a_axis * post;
    std::memcpy(out + o_off_b, b + b_off,
                static_cast<size_t>(b_axis * post) * sizeof(float));
  }
}

// ============================================================================
// Embedding / One-hot
// ============================================================================

// ids(a1,a0) int32 -> out(a1,a0,emb_dim) where out is (o2,o1,o0) =
// (a1,a0,emb_dim)
inline void embedding(const int32_t* ids, int64_t id1, int64_t id0,
                      const float* emb, int64_t e1, int64_t e0, float* out,
                      int64_t o2, int64_t o1, int64_t o0) {
  assert(o2 == id1 && o1 == id0 && o0 == e0);
  for (int64_t i1 = 0; i1 < id1; ++i1) {
    for (int64_t i0 = 0; i0 < id0; ++i0) {
      const int32_t tok = ids[off_2d(i1, i0, id0)];
      assert(0 <= tok && tok < e1);
      const float* src = emb + static_cast<int64_t>(tok) * e0;
      float* dst = out + off_3d(i1, i0, 0, o1, o0);
      std::memcpy(dst, src, static_cast<size_t>(e0) * sizeof(float));
    }
  }
}

inline void one_hot(const int32_t* ids, int64_t n, int32_t num_classes,
                    float on, float off, float* out, int64_t o1, int64_t o0) {
  assert(o1 == n && o0 == num_classes);
  for (int64_t i = 0; i < n; ++i) {
    const int32_t c = ids[i];
    assert(0 <= c && c < num_classes);
    float* row = out + i * o0;
    for (int32_t j = 0; j < num_classes; ++j) row[j] = off;
    row[c] = on;
  }
}

// ============================================================================
// Norms: RMSNorm / LayerNorm (both along last dim a0)
// ============================================================================

inline void rms_norm_lastdim(const float* x, int64_t outer, int64_t a0,
                             const float* weight, int64_t w0, float eps,
                             float* out) {
  assert(a0 == w0);
  for (int64_t i = 0; i < outer; ++i) {
    const float* row = x + i * a0;
    double ms = 0.0;
    for (int64_t j = 0; j < a0; ++j) {
      const double v = static_cast<double>(row[j]);
      ms += v * v;
    }
    ms /= static_cast<double>(a0);
    const float inv = 1.0f / std::sqrt(static_cast<float>(ms) + eps);
    float* orow = out + i * a0;
    for (int64_t j = 0; j < a0; ++j) orow[j] = (row[j] * inv) * weight[j];
  }
}

inline void rms_norm(const float* x, int64_t x2, int64_t x1, int64_t x0,
                     const float* weight, int64_t w0, float eps, float* out,
                     int64_t o2, int64_t o1, int64_t o0) {
  assert(x2 == o2 && x1 == o1 && x0 == o0);
  rms_norm_lastdim(x, x2 * x1, x0, weight, w0, eps, out);
}

// LayerNorm: out = (x-mean)/sqrt(var+eps)*gamma + beta
inline void layer_norm_lastdim(const float* x, int64_t outer, int64_t a0,
                               const float* gamma, int64_t g0,
                               const float* beta, int64_t b0, float eps,
                               float* out) {
  assert(a0 == g0 && a0 == b0);
  for (int64_t i = 0; i < outer; ++i) {
    const float* row = x + i * a0;
    double mean = 0.0;
    for (int64_t j = 0; j < a0; ++j) mean += static_cast<double>(row[j]);
    mean /= static_cast<double>(a0);

    double var = 0.0;
    for (int64_t j = 0; j < a0; ++j) {
      const double d = static_cast<double>(row[j]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(a0);

    const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
    float* orow = out + i * a0;
    for (int64_t j = 0; j < a0; ++j) {
      const float xn =
          static_cast<float>((static_cast<double>(row[j]) - mean)) * inv;
      orow[j] = xn * gamma[j] + beta[j];
    }
  }
}

// ============================================================================
// RoPE / Rotary (Q,K are float, in-place)
// Q,K: (a3,a2,a1,a0) = (B,S,H,D), D must be even
// cos/sin: (S,D) => (c1,c0) = (a2,a0)
// ============================================================================

inline void apply_rotary_pos_emb_4d_inplace(
    float* q, int64_t q3, int64_t q2, int64_t q1, int64_t q0, float* k,
    int64_t k3, int64_t k2, int64_t k1, int64_t k0, const float* cos,
    int64_t c1, int64_t c0, const float* sin, int64_t s1, int64_t s0) {
  assert(q3 == k3 && q2 == k2 && q1 == k1 && q0 == k0);
  assert(c1 == q2 && c0 == q0);
  assert(s1 == q2 && s0 == q0);
  assert((q0 % 2) == 0);

  const int64_t B = q3, S = q2, H = q1, D = q0, half = D / 2;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < S; ++t) {
      const float* c = cos + t * D;
      const float* s = sin + t * D;
      for (int64_t h = 0; h < H; ++h) {
        float* qv = q + off_4d(b, t, h, 0, q2, q1, q0);
        float* kv = k + off_4d(b, t, h, 0, k2, k1, k0);
        for (int64_t i = 0; i < D; ++i) {
          const int64_t src = (i < half) ? (i + half) : (i - half);
          const float rot_q = (i < half) ? (-qv[src]) : qv[src];
          const float rot_k = (i < half) ? (-kv[src]) : kv[src];
          const float qi = qv[i];
          const float ki = kv[i];
          qv[i] = qi * c[i] + rot_q * s[i];
          kv[i] = ki * c[i] + rot_k * s[i];
        }
      }
    }
  }
}

// 5D variant: treat leading dims (q4,q3) as big batch, inner is (S,H,D)
inline void apply_rotary_pos_emb_5d_inplace(float* q, int64_t q4, int64_t q3,
                                            int64_t q2, int64_t q1, int64_t q0,
                                            float* k, int64_t k4, int64_t k3,
                                            int64_t k2, int64_t k1, int64_t k0,
                                            const float* cos, int64_t c1,
                                            int64_t c0, const float* sin,
                                            int64_t s1, int64_t s0) {
  assert(q4 == k4 && q3 == k3 && q2 == k2 && q1 == k1 && q0 == k0);
  const int64_t big_batch = q4 * q3;
  const int64_t inner = q2 * q1 * q0;
  for (int64_t bb = 0; bb < big_batch; ++bb) {
    apply_rotary_pos_emb_4d_inplace(q + bb * inner, 1, q2, q1, q0,
                                    k + bb * inner, 1, q2, q1, q0, cos, c1, c0,
                                    sin, s1, s0);
  }
}

// ============================================================================
// GQA/MQA: repeat_kv
// x: (B,S,KV,D) => (a3,a2,a1,a0)
// out: (B,S,KV*n_rep,D)
// ============================================================================

inline void repeat_kv_4d(const float* x, int64_t x3, int64_t x2, int64_t x1,
                         int64_t x0, int64_t n_rep, float* out, int64_t o3,
                         int64_t o2, int64_t o1, int64_t o0) {
  assert(o3 == x3 && o2 == x2 && o0 == x0);
  assert(o1 == x1 * n_rep);
  for (int64_t b = 0; b < x3; ++b) {
    for (int64_t s = 0; s < x2; ++s) {
      for (int64_t kv = 0; kv < x1; ++kv) {
        const float* src = x + off_4d(b, s, kv, 0, x2, x1, x0);
        for (int64_t r = 0; r < n_rep; ++r) {
          const int64_t h = kv * n_rep + r;
          float* dst = out + off_4d(b, s, h, 0, o2, o1, o0);
          std::memcpy(dst, src, static_cast<size_t>(x0) * sizeof(float));
        }
      }
    }
  }
}

// ============================================================================
// KV-cache concat along sequence dim (S is a2):
// past(B,P,H,D) + cur(B,C,H,D) -> out(B,P+C,H,D)
// dims: (a3,a2,a1,a0) = (B,S,H,D)
// ============================================================================

inline void concat_seq_4d(const float* past, int64_t p3, int64_t p2, int64_t p1,
                          int64_t p0, const float* cur, int64_t c3, int64_t c2,
                          int64_t c1, int64_t c0, float* out, int64_t o3,
                          int64_t o2, int64_t o1, int64_t o0) {
  assert(p3 == c3 && p1 == c1 && p0 == c0);
  assert(o3 == p3 && o2 == p2 + c2 && o1 == p1 && o0 == p0);

  const int64_t B = p3, H = p1, D = p0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < p2; ++s) {
      for (int64_t h = 0; h < H; ++h) {
        const float* src = past + off_4d(b, s, h, 0, p2, p1, p0);
        float* dst = out + off_4d(b, s, h, 0, o2, o1, o0);
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
      }
    }
    for (int64_t s = 0; s < c2; ++s) {
      for (int64_t h = 0; h < H; ++h) {
        const float* src = cur + off_4d(b, s, h, 0, c2, c1, c0);
        float* dst = out + off_4d(b, p2 + s, h, 0, o2, o1, o0);
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
      }
    }
  }
}

// ============================================================================
// Attention helpers: causal mask + padding mask
// scores: (B,H,M,N) => dims (s3,s2,s1,s0) with negative-index meaning:
//   s0 = N (last dim, contiguous), s1 = M, s2 = H, s3 = B
// ============================================================================

// Apply causal mask on the *last block* (typical when you append K/V cache):
// scores[..., (N-cur_len):N] add -inf above diagonal.
inline void apply_causal_mask_last_block(float* scores, int64_t s3, int64_t s2,
                                         int64_t s1, int64_t s0,
                                         int64_t cur_len) {
  // Here s1==M is the query length of current chunk (usually = cur_len).
  assert(s1 == cur_len);
  assert(s0 >= cur_len);

  const int64_t B = s3, H = s2, M = s1, N = s0;
  const int64_t col0 = N - cur_len;
  const float nin = neg_inf_f32();

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t i = 0; i < cur_len; ++i) {
        float* row = scores + off_4d(b, h, i, 0, H, M, N);
        for (int64_t j = i + 1; j < cur_len; ++j) row[col0 + j] += nin;
      }
    }
  }
}

// Add padding mask (uint8): mask(B,N), 1=keep, 0=mask
inline void add_attention_mask(float* scores, int64_t s3, int64_t s2,
                               int64_t s1, int64_t s0, const uint8_t* mask,
                               int64_t m1, int64_t m0, float neg = -1e9f) {
  assert(m1 == s3 && m0 == s0);
  const int64_t B = s3, H = s2, M = s1, N = s0;

  for (int64_t b = 0; b < B; ++b) {
    const uint8_t* mb = mask + b * N;
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t i = 0; i < M; ++i) {
        float* row = scores + off_4d(b, h, i, 0, H, M, N);
        for (int64_t j = 0; j < N; ++j) {
          if (mb[j] == 0) row[j] += neg;
        }
      }
    }
  }
}

// ============================================================================
// MoE / routing primitives (topk / normalize / bincount / argsort / gather /
// scatter_add)
// ============================================================================

// x: (rows, cols) => dims (x1,x0). TopK over last dim x0.
inline void topk_lastdim_2d(const float* x, int64_t x1, int64_t x0, int32_t k,
                            int32_t* out_idx, int64_t oi1, int64_t oi0,
                            float* out_val, int64_t ov1, int64_t ov0) {
  assert(oi1 == x1 && ov1 == x1);
  assert(oi0 == k && ov0 == k);
  assert(k > 0 && k <= x0);

  for (int64_t r = 0; r < x1; ++r) {
    const float* row = x + r * x0;
    int32_t* irow = out_idx + r * k;
    float* vrow = out_val + r * k;

    for (int32_t t = 0; t < k; ++t) {
      irow[t] = -1;
      vrow[t] = -std::numeric_limits<float>::infinity();
    }

    for (int32_t c = 0; c < static_cast<int32_t>(x0); ++c) {
      const float v = row[c];
      int32_t worst = 0;
      for (int32_t t = 1; t < k; ++t)
        if (vrow[t] < vrow[worst]) worst = t;
      if (v > vrow[worst]) {
        vrow[worst] = v;
        irow[worst] = c;
      }
    }
  }
}

inline void normalize_topk_prob_2d_inplace(float* w, int64_t w1, int64_t w0,
                                           float eps = 1e-20f) {
  for (int64_t r = 0; r < w1; ++r) {
    float* row = w + r * w0;
    double sum = 0.0;
    for (int64_t j = 0; j < w0; ++j) sum += static_cast<double>(row[j]);
    const float inv = static_cast<float>(1.0 / (sum + eps));
    for (int64_t j = 0; j < w0; ++j) row[j] *= inv;
  }
}

inline void bincount_i32(const int32_t* x, int64_t n, int32_t num_bins,
                         int64_t* out_counts, int64_t c0) {
  assert(c0 == num_bins);
  for (int32_t i = 0; i < num_bins; ++i) out_counts[i] = 0;
  for (int64_t i = 0; i < n; ++i) {
    const int32_t v = x[i];
    assert(0 <= v && v < num_bins);
    out_counts[v] += 1;
  }
}

inline void prefix_sum_i64_inplace(int64_t* x, int64_t n) {
  int64_t acc = 0;
  for (int64_t i = 0; i < n; ++i) {
    acc += x[i];
    x[i] = acc;
  }
}

// idxs initialized as 0..n-1, then sort by keys[idx] ascending.
inline void argsort_by_key_i32(int64_t* idxs, int64_t n, const int32_t* keys) {
  std::sort(idxs, idxs + n, [&](int64_t a, int64_t b) {
    const int32_t ka = keys[a];
    const int32_t kb = keys[b];
    return (ka < kb) || (ka == kb && a < b);
  });
}

// Gather rows: out(n,cols) = x[idxs[i], :]
inline void gather_rows_2d(const float* x, int64_t x1, int64_t x0,
                           const int64_t* idxs, int64_t n, float* out,
                           int64_t o1, int64_t o0) {
  assert(o1 == n && o0 == x0);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t r = idxs[i];
    assert(0 <= r && r < x1);
    std::memcpy(out + i * o0, x + r * x0,
                static_cast<size_t>(x0) * sizeof(float));
  }
}

// Scatter-add rows: dst[idxs[i], :] += src[i, :]
inline void scatter_add_rows_2d(float* dst, int64_t d1, int64_t d0,
                                const int64_t* idxs, int64_t n,
                                const float* src, int64_t s1, int64_t s0) {
  assert(s1 == n && s0 == d0);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t r = idxs[i];
    assert(0 <= r && r < d1);
    float* dd = dst + r * d0;
    const float* ss = src + i * s0;
    for (int64_t j = 0; j < d0; ++j) dd[j] += ss[j];
  }
}

// repeat_interleave over first dim for 2D: x(x1,x0) -> out(x1*rep,x0)
inline void repeat_interleave_rows_2d(const float* x, int64_t x1, int64_t x0,
                                      int32_t rep, float* out, int64_t o1,
                                      int64_t o0) {
  assert(rep > 0);
  assert(o1 == x1 * rep && o0 == x0);
  for (int64_t r = 0; r < x1; ++r) {
    const float* src = x + r * x0;
    for (int32_t k = 0; k < rep; ++k) {
      float* dst = out + (r * rep + k) * x0;
      std::memcpy(dst, src, static_cast<size_t>(x0) * sizeof(float));
    }
  }
}

// Multiply each row by a scalar weight: y[r,:] = x[r,:] * w[r]
inline void mul_rows_by_scalar_2d(const float* x, int64_t x1, int64_t x0,
                                  const float* w, int64_t w0, float* out,
                                  int64_t o1, int64_t o0) {
  assert(w0 == x1);
  assert(o1 == x1 && o0 == x0);
  for (int64_t r = 0; r < x1; ++r) {
    const float s = w[r];
    const float* src = x + r * x0;
    float* dst = out + r * x0;
    for (int64_t c = 0; c < x0; ++c) dst[c] = src[c] * s;
  }
}

// ============================================================================
// Where / Masked fill (1D, easy building block)
// cond: uint8 (0/1), pick a if cond else b
// ============================================================================

inline void where_1d(const uint8_t* cond, int64_t n, const float* a, int64_t na,
                     const float* b, int64_t nb, float* out, int64_t no) {
  assert(n == na && n == nb && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = cond[i] ? a[i] : b[i];
}

inline void masked_fill_1d(const float* x, int64_t n, const uint8_t* mask,
                           int64_t nm, float val, float* out, int64_t no) {
  assert(n == nm && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = mask[i] ? val : x[i];
}

// ============================================================================
// Quantization baseline (per-tensor int8)
// ============================================================================

inline void quantize_per_tensor_f32_to_i8(const float* x, int64_t n,
                                          float scale, int32_t zero_point,
                                          int8_t* out, int64_t no) {
  assert(n == no);
  assert(scale > 0.0f);
  for (int64_t i = 0; i < n; ++i) {
    const float q = x[i] / scale + static_cast<float>(zero_point);
    const int32_t qi = static_cast<int32_t>(std::nearbyint(q));
    const int32_t clamped = std::max(-128, std::min(127, qi));
    out[i] = static_cast<int8_t>(clamped);
  }
}

inline void dequantize_per_tensor_i8_to_f32(const int8_t* x, int64_t n,
                                            float scale, int32_t zero_point,
                                            float* out, int64_t no) {
  assert(n == no);
  for (int64_t i = 0; i < n; ++i) {
    const int32_t xi = static_cast<int32_t>(x[i]);
    out[i] = (static_cast<float>(xi - zero_point)) * scale;
  }
}

// ============================
// Extra: common transpose wrappers
// ============================

// (B,S,H,D) -> (B,H,S,D) : permute(0,2,1,3)
inline void transpose_bshd_to_bhsd(const float* x, int64_t x3, int64_t x2,
                                   int64_t x1, int64_t x0,  // (B,S,H,D)
                                   float* out, int64_t o3, int64_t o2,
                                   int64_t o1, int64_t o0) {
  // (B,H,S,D)
  assert(o3 == x3 && o2 == x1 && o1 == x2 && o0 == x0);
  const int64_t B = x3, S = x2, H = x1, D = x0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t h = 0; h < H; ++h) {
        const float* src = x + off_4d(b, s, h, 0, x2, x1, x0);  // (b,s,h,:)
        float* dst = out + off_4d(b, h, s, 0, o2, o1, o0);      // (b,h,s,:)
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
      }
    }
  }
}

// (B,H,S,D) -> (B,S,H,D) : permute(0,2,1,3)
inline void transpose_bhsd_to_bshd(const float* x, int64_t x3, int64_t x2,
                                   int64_t x1, int64_t x0,  // (B,H,S,D)
                                   float* out, int64_t o3, int64_t o2,
                                   int64_t o1, int64_t o0) {
  // (B,S,H,D)
  assert(o3 == x3 && o2 == x1 && o1 == x2 && o0 == x0);
  const int64_t B = x3, H = x2, S = x1, D = x0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t s = 0; s < S; ++s) {
        const float* src = x + off_4d(b, h, s, 0, x2, x1, x0);  // (b,h,s,:)
        float* dst = out + off_4d(b, s, h, 0, o2, o1, o0);      // (b,s,h,:)
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
      }
    }
  }
}

// (B,S,T) -> (B,T,S) : handy for some score layouts
inline void transpose_bst_to_bts(const float* x, int64_t x2, int64_t x1,
                                 int64_t x0,  // (B,S,T)
                                 float* out, int64_t o2, int64_t o1,
                                 int64_t o0) {
  // (B,T,S)
  assert(o2 == x2 && o1 == x0 && o0 == x1);
  const int64_t B = x2, S = x1, T = x0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t t = 0; t < T; ++t) {
        out[off_3d(b, t, s, o1, o0)] = x[off_3d(b, s, t, x1, x0)];
      }
    }
  }
}

// ============================
// Extra: attention-special matmul
// Shapes are *explicit* and follow negative-index convention.
// ============================

// q: (B,H,S,D)  => (q3,q2,q1,q0) = (B,H,S,D)
// k: (B,H,T,D)  => (k3,k2,k1,k0) = (B,H,T,D)
// out scores: (B,H,S,T) => (o3,o2,o1,o0) = (B,H,S,T)
// scale is typically 1/sqrt(D) (pass 1.0f if you already scaled elsewhere)
inline void attn_qk_matmul(const float* q, int64_t q3, int64_t q2, int64_t q1,
                           int64_t q0, const float* k, int64_t k3, int64_t k2,
                           int64_t k1, int64_t k0, float scale, float* out,
                           int64_t o3, int64_t o2, int64_t o1, int64_t o0) {
  assert(q3 == k3 && q2 == k2 && q0 == k0);
  assert(o3 == q3 && o2 == q2 && o1 == q1 && o0 == k1);
  const int64_t B = q3, H = q2, S = q1, D = q0, T = k1;

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t s = 0; s < S; ++s) {
        const float* qv = q + off_4d(b, h, s, 0, q2, q1, q0);  // (D)
        float* row = out + off_4d(b, h, s, 0, o2, o1, o0);     // (T)
        for (int64_t t = 0; t < T; ++t) {
          const float* kv = k + off_4d(b, h, t, 0, k2, k1, k0);  // (D)
          double acc = 0.0;
          for (int64_t d = 0; d < D; ++d)
            acc += static_cast<double>(qv[d]) * static_cast<double>(kv[d]);
          row[t] = static_cast<float>(acc) * scale;
        }
      }
    }
  }
}

// probs: (B,H,S,T) => (p3,p2,p1,p0)=(B,H,S,T)
// v:     (B,H,T,D) => (v3,v2,v1,v0)=(B,H,T,D)
// out:   (B,H,S,D) => (o3,o2,o1,o0)=(B,H,S,D)
inline void attn_pv_matmul(const float* probs, int64_t p3, int64_t p2,
                           int64_t p1, int64_t p0, const float* v, int64_t v3,
                           int64_t v2, int64_t v1, int64_t v0, float* out,
                           int64_t o3, int64_t o2, int64_t o1, int64_t o0) {
  assert(p3 == v3 && p2 == v2 && p0 == v1);
  assert(o3 == p3 && o2 == p2 && o1 == p1 && o0 == v0);
  const int64_t B = p3, H = p2, S = p1, T = p0, D = v0;

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t s = 0; s < S; ++s) {
        const float* pr = probs + off_4d(b, h, s, 0, p2, p1, p0);  // (T)
        float* ov = out + off_4d(b, h, s, 0, o2, o1, o0);          // (D)
        // zero init
        for (int64_t d = 0; d < D; ++d) ov[d] = 0.0f;

        for (int64_t t = 0; t < T; ++t) {
          const float wt = pr[t];
          const float* vv = v + off_4d(b, h, t, 0, v2, v1, v0);  // (D)
          for (int64_t d = 0; d < D; ++d) ov[d] += wt * vv[d];
        }
      }
    }
  }
}

// scores: (B,H,S,T) softmax over last dim T
inline void attn_softmax_scores(const float* scores, int64_t s3, int64_t s2,
                                int64_t s1, int64_t s0, float* out, int64_t o3,
                                int64_t o2, int64_t o1, int64_t o0) {
  assert(o3 == s3 && o2 == s2 && o1 == s1 && o0 == s0);
  // outer = B*H*S, last dim = T
  softmax_lastdim(scores, s3 * s2 * s1, s0, out);
}

// ============================
// Extra: CE / NLL (float32 logits / log_probs)
// ============================

// logits: (N,C) => (l1,l0)=(N,C)
// target: (N)   => (t0) = N
// out: (1) scalar mean over non-ignored targets
inline void cross_entropy_logits_2d_mean(const float* logits, int64_t l1,
                                         int64_t l0, const int32_t* target,
                                         int64_t t0, int32_t ignore_index,
                                         float* out, int64_t o0) {
  assert(t0 == l1);
  assert(o0 == 1);

  double loss_sum = 0.0;
  int64_t count = 0;

  for (int64_t i = 0; i < l1; ++i) {
    const int32_t y = target[i];
    if (y == ignore_index) continue;
    assert(0 <= y && y < l0);

    const float* row = logits + i * l0;
    float mx = row[0];
    for (int64_t c = 1; c < l0; ++c) mx = std::max(mx, row[c]);

    double sum = 0.0;
    for (int64_t c = 0; c < l0; ++c)
      sum += static_cast<double>(std::exp(row[c] - mx));
    const double logsum = (sum > 0.0) ? std::log(sum) : 0.0;

    const double logp =
        static_cast<double>(row[y] - mx) - logsum;  // log-softmax
    loss_sum += -logp;
    ++count;
  }

  out[0] = (count > 0)
               ? static_cast<float>(loss_sum / static_cast<double>(count))
               : 0.0f;
}

// out: (1) scalar sum over non-ignored targets
inline void cross_entropy_logits_2d_sum(const float* logits, int64_t l1,
                                        int64_t l0, const int32_t* target,
                                        int64_t t0, int32_t ignore_index,
                                        float* out, int64_t o0) {
  assert(t0 == l1);
  assert(o0 == 1);

  double loss_sum = 0.0;
  for (int64_t i = 0; i < l1; ++i) {
    const int32_t y = target[i];
    if (y == ignore_index) continue;
    assert(0 <= y && y < l0);

    const float* row = logits + i * l0;
    float mx = row[0];
    for (int64_t c = 1; c < l0; ++c) mx = std::max(mx, row[c]);

    double sum = 0.0;
    for (int64_t c = 0; c < l0; ++c)
      sum += static_cast<double>(std::exp(row[c] - mx));
    const double logsum = (sum > 0.0) ? std::log(sum) : 0.0;

    const double logp = static_cast<double>(row[y] - mx) - logsum;
    loss_sum += -logp;
  }
  out[0] = static_cast<float>(loss_sum);
}

// per-sample loss: out(N)
inline void cross_entropy_logits_2d_none(const float* logits, int64_t l1,
                                         int64_t l0, const int32_t* target,
                                         int64_t t0, int32_t ignore_index,
                                         float* out, int64_t o0) {
  assert(t0 == l1);
  assert(o0 == l1);

  for (int64_t i = 0; i < l1; ++i) {
    const int32_t y = target[i];
    if (y == ignore_index) {
      out[i] = 0.0f;
      continue;
    }
    assert(0 <= y && y < l0);

    const float* row = logits + i * l0;
    float mx = row[0];
    for (int64_t c = 1; c < l0; ++c) mx = std::max(mx, row[c]);

    double sum = 0.0;
    for (int64_t c = 0; c < l0; ++c)
      sum += static_cast<double>(std::exp(row[c] - mx));
    const double logsum = (sum > 0.0) ? std::log(sum) : 0.0;

    const double logp = static_cast<double>(row[y] - mx) - logsum;
    out[i] = static_cast<float>(-logp);
  }
}

// log_probs: (N,C) already log-softmaxed
inline void nll_loss_log_probs_2d_mean(const float* log_probs, int64_t l1,
                                       int64_t l0, const int32_t* target,
                                       int64_t t0, int32_t ignore_index,
                                       float* out, int64_t o0) {
  assert(t0 == l1);
  assert(o0 == 1);

  double loss_sum = 0.0;
  int64_t count = 0;
  for (int64_t i = 0; i < l1; ++i) {
    const int32_t y = target[i];
    if (y == ignore_index) continue;
    assert(0 <= y && y < l0);
    loss_sum += -static_cast<double>(log_probs[i * l0 + y]);
    ++count;
  }
  out[0] = (count > 0)
               ? static_cast<float>(loss_sum / static_cast<double>(count))
               : 0.0f;
}

// LM-friendly shifted CE:
// logits: (B,S,V) => (l2,l1,l0) = (B,S,V)
// labels: (B,S)   => (y1,y0)   = (B,S)
// loss computed on logits[:,0:S-1,:] vs labels[:,1:S]
inline void cross_entropy_logits_3d_shifted_mean(
    const float* logits, int64_t l2, int64_t l1, int64_t l0,
    const int32_t* labels, int64_t y1, int64_t y0, int32_t ignore_index,
    float* out, int64_t o0) {
  assert(y1 == l2 && y0 == l1);
  assert(o0 == 1);
  const int64_t B = l2, S = l1, V = l0;
  if (S <= 1) {
    out[0] = 0.0f;
    return;
  }

  double loss_sum = 0.0;
  int64_t count = 0;

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S - 1; ++s) {
      const int32_t y = labels[off_2d(b, s + 1, y0)];
      if (y == ignore_index) continue;
      assert(0 <= y && y < V);

      const float* row = logits + off_3d(b, s, 0, l1, l0);  // logits[b,s,:]
      float mx = row[0];
      for (int64_t c = 1; c < V; ++c) mx = std::max(mx, row[c]);

      double sum = 0.0;
      for (int64_t c = 0; c < V; ++c)
        sum += static_cast<double>(std::exp(row[c] - mx));
      const double logsum = (sum > 0.0) ? std::log(sum) : 0.0;

      const double logp = static_cast<double>(row[y] - mx) - logsum;
      loss_sum += -logp;
      ++count;
    }
  }

  out[0] = (count > 0)
               ? static_cast<float>(loss_sum / static_cast<double>(count))
               : 0.0f;
}

// ============================
// Extra: view/reshape (no-op, just checks)
// ============================

inline void view_reshape_check_numel(int64_t in_numel, int64_t out_numel) {
  assert(in_numel == out_numel);
}

inline const float* view_reshape_2d_to_4d(const float* x, int64_t x1,
                                          int64_t x0, int64_t o3, int64_t o2,
                                          int64_t o1, int64_t o0) {
  view_reshape_check_numel(x1 * x0, o3 * o2 * o1 * o0);
  return x;
}

inline float* view_reshape_2d_to_4d(float* x, int64_t x1, int64_t x0,
                                    int64_t o3, int64_t o2, int64_t o1,
                                    int64_t o0) {
  view_reshape_check_numel(x1 * x0, o3 * o2 * o1 * o0);
  return x;
}

inline const float* view_reshape_3d_to_4d(const float* x, int64_t x2,
                                          int64_t x1, int64_t x0, int64_t o3,
                                          int64_t o2, int64_t o1, int64_t o0) {
  view_reshape_check_numel(x2 * x1 * x0, o3 * o2 * o1 * o0);
  return x;
}

inline float* view_reshape_3d_to_4d(float* x, int64_t x2, int64_t x1,
                                    int64_t x0, int64_t o3, int64_t o2,
                                    int64_t o1, int64_t o0) {
  view_reshape_check_numel(x2 * x1 * x0, o3 * o2 * o1 * o0);
  return x;
}

inline const float* view_reshape_3d_to_5d(const float* x, int64_t x2,
                                          int64_t x1, int64_t x0, int64_t o4,
                                          int64_t o3, int64_t o2, int64_t o1,
                                          int64_t o0) {
  view_reshape_check_numel(x2 * x1 * x0, o4 * o3 * o2 * o1 * o0);
  return x;
}

inline float* view_reshape_3d_to_5d(float* x, int64_t x2, int64_t x1,
                                    int64_t x0, int64_t o4, int64_t o3,
                                    int64_t o2, int64_t o1, int64_t o0) {
  view_reshape_check_numel(x2 * x1 * x0, o4 * o3 * o2 * o1 * o0);
  return x;
}

inline const float* view_reshape_4d_to_3d(const float* x, int64_t x3,
                                          int64_t x2, int64_t x1, int64_t x0,
                                          int64_t o2, int64_t o1, int64_t o0) {
  view_reshape_check_numel(x3 * x2 * x1 * x0, o2 * o1 * o0);
  return x;
}

inline float* view_reshape_4d_to_3d(float* x, int64_t x3, int64_t x2,
                                    int64_t x1, int64_t x0, int64_t o2,
                                    int64_t o1, int64_t o0) {
  view_reshape_check_numel(x3 * x2 * x1 * x0, o2 * o1 * o0);
  return x;
}

// Flatten all leading dims into one "outer", keep last dim (a0) unchanged.
// x: (..., a0) -> out: (outer, a0) (view/no-op)
inline const float* view_flatten_to_2d_keep_last(const float* x, int64_t numel,
                                                 int64_t a0, int64_t* out1,
                                                 int64_t* out0) {
  assert(a0 > 0);
  assert(numel % a0 == 0);
  *out1 = numel / a0;
  *out0 = a0;
  return x;
}

// ============================
// Extra: fused/utility elementwise
// ============================

inline void add_scaled(const float* a, int64_t n, const float* b, int64_t nb,
                       float alpha, float* out, int64_t no) {
  assert(n == nb && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] + alpha * b[i];
}

inline void fma3(const float* a, int64_t n, const float* b, int64_t nb,
                 const float* c, int64_t nc, float* out, int64_t no) {
  assert(n == nb && n == nc && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] * b[i] + c[i];
}

inline void add_inplace(float* a, int64_t n, const float* b, int64_t nb,
                        float alpha) {
  assert(n == nb);
  for (int64_t i = 0; i < n; ++i) a[i] += alpha * b[i];
}

// ============================
// Extra: bias add (last-dim) + optional in-place
// ============================

inline void add_bias_lastdim_inplace(float* x, int64_t outer, int64_t a0,
                                     const float* bias, int64_t b0) {
  assert(a0 == b0);
  for (int64_t i = 0; i < outer; ++i) {
    float* row = x + i * a0;
    for (int64_t j = 0; j < a0; ++j) row[j] += bias[j];
  }
}

inline void add_bias_lastdim(const float* x, int64_t outer, int64_t a0,
                             const float* bias, int64_t b0, float* out) {
  assert(a0 == b0);
  for (int64_t i = 0; i < outer; ++i) {
    const float* xr = x + i * a0;
    float* yr = out + i * a0;
    for (int64_t j = 0; j < a0; ++j) yr[j] = xr[j] + bias[j];
  }
}

// ============================
// Extra: dropout (deterministic xorshift64*), optional mask
// ============================

struct xorshift64s_state {
  uint64_t s = 88172645463325252ull;
};

inline uint64_t xorshift64s_next(xorshift64s_state* st) {
  uint64_t x = st->s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  st->s = x;
  return x * 2685821657736338717ull;
}

// uniform in [0,1)
inline float rand_uniform01(xorshift64s_state* st) {
  // Use top 24 bits to make a float in [0,1).
  const uint64_t r = xorshift64s_next(st);
  const uint32_t u = static_cast<uint32_t>(r >> 40);  // 24 bits
  return static_cast<float>(u) * (1.0f / 16777216.0f);
}

// dropout: out = x * mask / (1-p) ; mask=1 keep, 0 drop
inline void dropout(const float* x, int64_t n, float p, xorshift64s_state* rng,
                    float* out, int64_t no, uint8_t* mask_opt = nullptr,
                    int64_t m0 = 0) {
  assert(n == no);
  assert(p >= 0.0f && p < 1.0f);
  if (p == 0.0f || rng == nullptr) {
    std::memcpy(out, x, static_cast<size_t>(n) * sizeof(float));
    if (mask_opt != nullptr) {
      assert(m0 == n);
      std::memset(mask_opt, 1, static_cast<size_t>(n));
    }
    return;
  }
  if (mask_opt != nullptr) assert(m0 == n);

  const float keep = 1.0f - p;
  const float inv_keep = 1.0f / keep;

  for (int64_t i = 0; i < n; ++i) {
    const float u = rand_uniform01(rng);
    const uint8_t m =
        (u < keep) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    if (mask_opt != nullptr) mask_opt[i] = m;
    out[i] = m ? (x[i] * inv_keep) : 0.0f;
  }
}

// ============================
// Extra: linear (with bias) for common shapes
// Weight layout matches PyTorch nn.Linear: W(out, in) => (w1=out, w0=in)
// ============================

// x: (B,S,In) => (x2,x1,x0)
// w: (Out,In) => (w1,w0)
// b: (Out)    => (b0)
// out:(B,S,Out)=> (o2,o1,o0)
inline void linear(const float* x, int64_t x2, int64_t x1, int64_t x0,
                   const float* w, int64_t w1, int64_t w0, const float* b,
                   int64_t b0, float* out, int64_t o2, int64_t o1, int64_t o0) {
  assert(x2 == o2 && x1 == o1);
  assert(w0 == x0);
  assert(o0 == w1);
  assert(b0 == w1);

  const int64_t outer = x2 * x1;
  // out_flat(outer, out) = x_flat(outer, in) @ w(out,in)^T
  matmul_nt(x, outer, x0, w, w1, w0, out, outer, w1);
  add_bias_lastdim_inplace(out, outer, w1, b, b0);
}

// 2D linear: x(N,In) -> out(N,Out)
inline void linear(const float* x, int64_t x1, int64_t x0, const float* w,
                   int64_t w1, int64_t w0, const float* b, int64_t b0,
                   float* out, int64_t o1, int64_t o0) {
  assert(x1 == o1);
  assert(w0 == x0);
  assert(o0 == w1);
  assert(b0 == w1);

  matmul_nt(x, x1, x0, w, w1, w0, out, o1, o0);
  add_bias_lastdim_inplace(out, x1, o0, b, b0);
}

// ============================
// Extra: attention causal mask (full, not only last block)
// scores: (B,H,S,T) => (s3,s2,s1,s0)=(B,H,S,T)
// If T==S and it's self-attn, this is standard causal.
// If T>S (cache), still masks positions j > (prefix_offset + i).
// prefix_offset tells where current query block starts in the key timeline.
// Typical:
//   - no cache: prefix_offset=0, S=T
//   - cache: keys length = prefix_offset + S, so T == prefix_offset+S,
//   prefix_offset = past_len
inline void apply_causal_mask(float* scores, int64_t s3, int64_t s2, int64_t s1,
                              int64_t s0, int64_t prefix_offset) {
  const int64_t B = s3, H = s2, S = s1, T = s0;
  const float nin = neg_inf_f32();

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t i = 0; i < S; ++i) {
        // Allowed keys: [0 .. prefix_offset+i]
        const int64_t max_j = std::min<int64_t>(T - 1, prefix_offset + i);
        float* row = scores + off_4d(b, h, i, 0, H, S, T);
        for (int64_t j = max_j + 1; j < T; ++j) row[j] += nin;
      }
    }
  }
}

// ============================
// Extra: score scaling (in-place): scores *= scale
// scores: (B,H,S,T)
// ============================

inline void scale_scores_inplace(float* scores, int64_t s3, int64_t s2,
                                 int64_t s1, int64_t s0, float scale) {
  const int64_t n = s3 * s2 * s1 * s0;
  scale_inplace(scores, n, scale);
}

// ============================
// Extra: argmax / gather for logits
// ============================

// logits: (N,V) -> out_idx: (N)
inline void argmax_lastdim_2d(const float* x, int64_t x1, int64_t x0,
                              int32_t* out_idx, int64_t o0) {
  assert(o0 == x1);
  for (int64_t i = 0; i < x1; ++i) {
    const float* row = x + i * x0;
    int64_t best = 0;
    float bestv = row[0];
    for (int64_t j = 1; j < x0; ++j) {
      if (row[j] > bestv) {
        bestv = row[j];
        best = j;
      }
    }
    out_idx[i] = static_cast<int32_t>(best);
  }
}

// gather last-dim: x(N,V), idx(N) -> out(N)
inline void gather_lastdim_2d(const float* x, int64_t x1, int64_t x0,
                              const int32_t* idx, int64_t i0, float* out,
                              int64_t o0) {
  assert(i0 == x1);
  assert(o0 == x1);
  for (int64_t i = 0; i < x1; ++i) {
    const int32_t c = idx[i];
    assert(0 <= c && c < x0);
    out[i] = x[i * x0 + c];
  }
}

// ============================
// Extra: common reshape+transpose helpers for attention pipeline
// ============================

// After q_proj: x(B,S,H*D) => q(B,S,H,D) view
inline const float* view_bshd_from_bshd_flat(const float* x, int64_t B,
                                             int64_t S, int64_t HD, int64_t H,
                                             int64_t D) {
  assert(HD == H * D);
  return view_reshape_3d_to_4d(x, B, S, HD, B, S, H, D);
}

inline float* view_bshd_from_bshd_flat(float* x, int64_t B, int64_t S,
                                       int64_t HD, int64_t H, int64_t D) {
  assert(HD == H * D);
  return view_reshape_3d_to_4d(x, B, S, HD, B, S, H, D);
}

// Merge heads: x(B,S,H,D) -> out(B,S,H*D) view
inline const float* view_bshd_flat_from_bshd(const float* x, int64_t B,
                                             int64_t S, int64_t H, int64_t D,
                                             int64_t* out_hd) {
  *out_hd = H * D;
  return view_reshape_4d_to_3d(x, B, S, H, D, B, S, *out_hd);
}

inline float* view_bshd_flat_from_bshd(float* x, int64_t B, int64_t S,
                                       int64_t H, int64_t D, int64_t* out_hd) {
  *out_hd = H * D;
  return view_reshape_4d_to_3d(x, B, S, H, D, B, S, *out_hd);
}

// ============================
// Extra: attention "full" wrapper: scores -> softmax -> out
// (You can still call the primitives individually; this just reduces
// boilerplate.) q: (B,H,S,D), k: (B,H,T,D), v: (B,H,T,D) mask: (B,T) uint8
// 1=keep 0=mask, optional (pass nullptr if none) prefix_offset: past length for
// cache (see apply_causal_mask) out: (B,H,S,D) temps: probs(B,H,S,T) caller
// must allocate
// ============================

inline void attention_core(const float* q, int64_t q3, int64_t q2, int64_t q1,
                           int64_t q0, const float* k, int64_t k3, int64_t k2,
                           int64_t k1, int64_t k0, const float* v, int64_t v3,
                           int64_t v2, int64_t v1, int64_t v0,
                           const uint8_t* attn_mask_bt, int64_t m1, int64_t m0,
                           int64_t prefix_offset, float* scores, int64_t s3,
                           int64_t s2, int64_t s1, int64_t s0, float* probs,
                           int64_t p3, int64_t p2, int64_t p1, int64_t p0,
                           float* out, int64_t o3, int64_t o2, int64_t o1,
                           int64_t o0) {
  // 1) scores = q @ k^T
  const float scale = 1.0f / std::sqrt(static_cast<float>(q0));
  attn_qk_matmul(q, q3, q2, q1, q0, k, k3, k2, k1, k0, scale, scores, s3, s2,
                 s1, s0);

  // 2) causal
  apply_causal_mask(scores, s3, s2, s1, s0, prefix_offset);

  // 3) padding mask
  if (attn_mask_bt != nullptr) {
    assert(m1 == s3 && m0 == s0);
    add_attention_mask(scores, s3, s2, s1, s0, attn_mask_bt, m1, m0);
  }

  // 4) softmax over last dim T
  attn_softmax_scores(scores, s3, s2, s1, s0, probs, p3, p2, p1, p0);

  // 5) out = probs @ v
  attn_pv_matmul(probs, p3, p2, p1, p0, v, v3, v2, v1, v0, out, o3, o2, o1, o0);
}

// ============================
// Extra: slice/concat/split along last dim (a0) for 2D/3D/4D/5D
// This is the most common "hidden split/merge" pattern.
// ============================

// out = x[..., start : start+len]  (copy-based)
inline void slice_lastdim_2d(const float* x, int64_t x1, int64_t x0,
                             int64_t start, int64_t len, float* out, int64_t o1,
                             int64_t o0) {
  assert(o1 == x1 && o0 == len);
  assert(0 <= start && start + len <= x0);
  for (int64_t i1 = 0; i1 < x1; ++i1) {
    const float* src = x + i1 * x0 + start;
    float* dst = out + i1 * o0;
    std::memcpy(dst, src, static_cast<size_t>(len) * sizeof(float));
  }
}

inline void slice_lastdim_3d(const float* x, int64_t x2, int64_t x1, int64_t x0,
                             int64_t start, int64_t len, float* out, int64_t o2,
                             int64_t o1, int64_t o0) {
  assert(o2 == x2 && o1 == x1 && o0 == len);
  assert(0 <= start && start + len <= x0);
  const int64_t outer = x2 * x1;
  slice_lastdim_2d(x, outer, x0, start, len, out, outer, len);
}

inline void slice_lastdim_4d(const float* x, int64_t x3, int64_t x2, int64_t x1,
                             int64_t x0, int64_t start, int64_t len, float* out,
                             int64_t o3, int64_t o2, int64_t o1, int64_t o0) {
  assert(o3 == x3 && o2 == x2 && o1 == x1 && o0 == len);
  assert(0 <= start && start + len <= x0);
  const int64_t outer = x3 * x2 * x1;
  slice_lastdim_2d(x, outer, x0, start, len, out, outer, len);
}

inline void slice_lastdim_5d(const float* x, int64_t x4, int64_t x3, int64_t x2,
                             int64_t x1, int64_t x0, int64_t start, int64_t len,
                             float* out, int64_t o4, int64_t o3, int64_t o2,
                             int64_t o1, int64_t o0) {
  assert(o4 == x4 && o3 == x3 && o2 == x2 && o1 == x1 && o0 == len);
  assert(0 <= start && start + len <= x0);
  const int64_t outer = x4 * x3 * x2 * x1;
  slice_lastdim_2d(x, outer, x0, start, len, out, outer, len);
}

// out = concat(x, y) along last dim
inline void concat_lastdim_2d(const float* a, int64_t a1, int64_t a0,
                              const float* b, int64_t b1, int64_t b0,
                              float* out, int64_t o1, int64_t o0) {
  assert(a1 == b1);
  assert(o1 == a1);
  assert(o0 == a0 + b0);
  for (int64_t i1 = 0; i1 < a1; ++i1) {
    float* dst = out + i1 * o0;
    std::memcpy(dst, a + i1 * a0, static_cast<size_t>(a0) * sizeof(float));
    std::memcpy(dst + a0, b + i1 * b0, static_cast<size_t>(b0) * sizeof(float));
  }
}

inline void concat_lastdim_3d(const float* a, int64_t a2, int64_t a1,
                              int64_t a0, const float* b, int64_t b2,
                              int64_t b1, int64_t b0, float* out, int64_t o2,
                              int64_t o1, int64_t o0) {
  assert(a2 == b2 && a1 == b1);
  assert(o2 == a2 && o1 == a1);
  assert(o0 == a0 + b0);
  const int64_t outer = a2 * a1;
  concat_lastdim_2d(a, outer, a0, b, outer, b0, out, outer, o0);
}

inline void concat_lastdim_4d(const float* a, int64_t a3, int64_t a2,
                              int64_t a1, int64_t a0, const float* b,
                              int64_t b3, int64_t b2, int64_t b1, int64_t b0,
                              float* out, int64_t o3, int64_t o2, int64_t o1,
                              int64_t o0) {
  assert(a3 == b3 && a2 == b2 && a1 == b1);
  assert(o3 == a3 && o2 == a2 && o1 == a1);
  assert(o0 == a0 + b0);
  const int64_t outer = a3 * a2 * a1;
  concat_lastdim_2d(a, outer, a0, b, outer, b0, out, outer, o0);
}

inline void concat_lastdim_5d(const float* a, int64_t a4, int64_t a3,
                              int64_t a2, int64_t a1, int64_t a0,
                              const float* b, int64_t b4, int64_t b3,
                              int64_t b2, int64_t b1, int64_t b0, float* out,
                              int64_t o4, int64_t o3, int64_t o2, int64_t o1,
                              int64_t o0) {
  assert(a4 == b4 && a3 == b3 && a2 == b2 && a1 == b1);
  assert(o4 == a4 && o3 == a3 && o2 == a2 && o1 == a1);
  assert(o0 == a0 + b0);
  const int64_t outer = a4 * a3 * a2 * a1;
  concat_lastdim_2d(a, outer, a0, b, outer, b0, out, outer, o0);
}

// Split last dim into 3 chunks: out0/out1/out2
inline void split_lastdim_2d_3way(const float* x, int64_t x1, int64_t x0,
                                  int64_t a0_len, int64_t b0_len,
                                  int64_t c0_len, float* out0, int64_t o01,
                                  int64_t o00, float* out1, int64_t o11,
                                  int64_t o10, float* out2, int64_t o21,
                                  int64_t o20) {
  assert(a0_len + b0_len + c0_len == x0);
  slice_lastdim_2d(x, x1, x0, 0, a0_len, out0, o01, o00);
  slice_lastdim_2d(x, x1, x0, a0_len, b0_len, out1, o11, o10);
  slice_lastdim_2d(x, x1, x0, a0_len + b0_len, c0_len, out2, o21, o20);
}

// ============================
// Extra: fused QKV linear (common in transformer)
// x: (B,S,In)
// w_qkv: (3*Out, In)  where Out = H*D (or KV*D etc), layout (w1,w0)=(3*Out,In)
// b_qkv: (3*Out)
// out_qkv: (B,S,3*Out)
// Then you can slice_lastdim_* to get q,k,v.
// ============================

inline void linear_qkv_fused(const float* x, int64_t x2, int64_t x1,
                             int64_t x0,  // (B,S,In)
                             const float* w_qkv, int64_t w1,
                             int64_t w0,                      // (3*Out,In)
                             const float* b_qkv, int64_t b0,  // (3*Out)
                             float* out, int64_t o2, int64_t o1, int64_t o0) {
  // (B,S,3*Out)
  assert(x2 == o2 && x1 == o1);
  assert(w0 == x0);
  assert(o0 == w1);
  assert(b0 == w1);
  // out = x @ w^T + b
  linear(x, x2, x1, x0, w_qkv, w1, w0, b_qkv, b0, out, o2, o1, o0);
}

// Convenience: produce q/k/v directly into separate outputs (copy-based split).
// out_q/out_k/out_v are (B,S,Out)
inline void linear_qkv_fused_split3(
    const float* x, int64_t B, int64_t S, int64_t In, const float* w_qkv,
    int64_t w1, int64_t w0, const float* b_qkv, int64_t b0, float* tmp_qkv,
    int64_t t2, int64_t t1, int64_t t0,                // (B,S,3*Out)
    float* out_q, int64_t q2, int64_t q1, int64_t q0,  // (B,S,Out)
    float* out_k, int64_t k2, int64_t k1, int64_t k0, float* out_v, int64_t v2,
    int64_t v1, int64_t v0) {
  assert(B == t2 && S == t1);
  assert(B == q2 && S == q1);
  assert(B == k2 && S == k1);
  assert(B == v2 && S == v1);
  assert(w0 == In);
  assert(b0 == w1);
  const int64_t Out3 = w1;
  assert(t0 == Out3);
  assert(Out3 % 3 == 0);
  const int64_t Out = Out3 / 3;
  assert(q0 == Out && k0 == Out && v0 == Out);

  linear_qkv_fused(x, B, S, In, w_qkv, w1, w0, b_qkv, b0, tmp_qkv, t2, t1, t0);
  slice_lastdim_3d(tmp_qkv, B, S, Out3, 0, Out, out_q, q2, q1, q0);
  slice_lastdim_3d(tmp_qkv, B, S, Out3, Out, Out, out_k, k2, k1, k0);
  slice_lastdim_3d(tmp_qkv, B, S, Out3, 2 * Out, Out, out_v, v2, v1, v0);
}

// ============================
// Extra: label-smoothing CrossEntropy (logits)
// logits: (N,C)
// target: (N)
// ignore_index supported
// smoothing in [0,1)
// loss per token: (1-eps)*(-logp_y) + eps*(-mean_logp)
// reduction: mean over non-ignored
// ============================

inline void cross_entropy_logits_2d_label_smoothing_mean(
    const float* logits, int64_t l1, int64_t l0, const int32_t* target,
    int64_t t0, int32_t ignore_index, float smoothing, float* out, int64_t o0) {
  assert(t0 == l1);
  assert(o0 == 1);
  assert(smoothing >= 0.0f && smoothing < 1.0f);

  double loss_sum = 0.0;
  int64_t count = 0;

  for (int64_t i = 0; i < l1; ++i) {
    const int32_t y = target[i];
    if (y == ignore_index) continue;
    assert(0 <= y && y < l0);

    const float* row = logits + i * l0;

    // stable log-softmax stats
    float mx = row[0];
    for (int64_t c = 1; c < l0; ++c) mx = std::max(mx, row[c]);

    double sum = 0.0;
    for (int64_t c = 0; c < l0; ++c)
      sum += static_cast<double>(std::exp(row[c] - mx));
    const double logsum = (sum > 0.0) ? std::log(sum) : 0.0;

    // logp_y
    const double logp_y = static_cast<double>(row[y] - mx) - logsum;

    // mean logp over all classes: mean_c logp_c
    double mean_logp = 0.0;
    for (int64_t c = 0; c < l0; ++c) {
      const double logp_c = static_cast<double>(row[c] - mx) - logsum;
      mean_logp += logp_c;
    }
    mean_logp /= static_cast<double>(l0);

    const double nll = -logp_y;
    const double smooth = -mean_logp;
    const double loss = (1.0 - static_cast<double>(smoothing)) * nll +
                        static_cast<double>(smoothing) * smooth;

    loss_sum += loss;
    ++count;
  }

  out[0] = (count > 0)
               ? static_cast<float>(loss_sum / static_cast<double>(count))
               : 0.0f;
}

// ============================
// Extra: logits -> log_softmax (2D/3D), and NLL on shifted LM logits
// ============================

inline void log_softmax_2d(const float* x, int64_t x1, int64_t x0, float* out,
                           int64_t o1, int64_t o0) {
  assert(o1 == x1 && o0 == x0);
  log_softmax_lastdim(x, x1, x0, out);
}

inline void log_softmax_3d(const float* x, int64_t x2, int64_t x1, int64_t x0,
                           float* out, int64_t o2, int64_t o1, int64_t o0) {
  assert(o2 == x2 && o1 == x1 && o0 == x0);
  log_softmax_lastdim(x, x2 * x1, x0, out);
}

// NLL for LM with precomputed log_probs: log_probs(B,S,V), labels(B,S), shift
// like HF. loss mean over valid tokens.
inline void nll_loss_log_probs_3d_shifted_mean(
    const float* log_probs, int64_t l2, int64_t l1, int64_t l0,
    const int32_t* labels, int64_t y1, int64_t y0, int32_t ignore_index,
    float* out, int64_t o0) {
  assert(y1 == l2 && y0 == l1);
  assert(o0 == 1);
  const int64_t B = l2, S = l1, V = l0;
  if (S <= 1) {
    out[0] = 0.0f;
    return;
  }

  double loss_sum = 0.0;
  int64_t count = 0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S - 1; ++s) {
      const int32_t y = labels[off_2d(b, s + 1, y0)];
      if (y == ignore_index) continue;
      assert(0 <= y && y < V);
      const float lp = log_probs[off_3d(b, s, y, l1, l0)];
      loss_sum += -static_cast<double>(lp);
      ++count;
    }
  }
  out[0] = (count > 0)
               ? static_cast<float>(loss_sum / static_cast<double>(count))
               : 0.0f;
}

// ============================
// Extra: mask utilities for attention (build/convert)
// ============================

// Convert float mask (0/1 or >0.5) to uint8 (1 keep, 0 mask)
inline void mask_f32_to_u8_keep(const float* m, int64_t n, float thr,
                                uint8_t* out, int64_t o0) {
  assert(o0 == n);
  for (int64_t i = 0; i < n; ++i)
    out[i] = (m[i] > thr) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
}

// Build causal mask as uint8 keep matrix (S,T): keep if j<=prefix+i
inline void build_causal_keep_mask_2d(int64_t S, int64_t T,
                                      int64_t prefix_offset, uint8_t* out,
                                      int64_t o1, int64_t o0) {
  assert(o1 == S && o0 == T);
  for (int64_t i = 0; i < S; ++i) {
    const int64_t max_j = std::min<int64_t>(T - 1, prefix_offset + i);
    uint8_t* row = out + i * T;
    for (int64_t j = 0; j < T; ++j)
      row[j] = (j <= max_j) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
  }
}

// Apply keep-mask (S,T) onto scores(B,H,S,T) by adding neg to masked cells.
// This is slower than add_attention_mask(B,T) but more general.
inline void add_score_mask_2d(float* scores, int64_t s3, int64_t s2, int64_t s1,
                              int64_t s0, const uint8_t* keep, int64_t k1,
                              int64_t k0, float neg = -1e9f) {
  assert(k1 == s1 && k0 == s0);
  const int64_t B = s3, H = s2, S = s1, T = s0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t i = 0; i < S; ++i) {
        float* row = scores + off_4d(b, h, i, 0, H, S, T);
        const uint8_t* km = keep + i * T;
        for (int64_t j = 0; j < T; ++j)
          if (!km[j]) row[j] += neg;
      }
    }
  }
}

// ============================
// Extra: tiny helpers: sqrt(head_dim) scaling, safe division
// ============================

inline float inv_sqrt_f32(int64_t d) {
  assert(d > 0);
  return 1.0f / std::sqrt(static_cast<float>(d));
}

inline void safe_div(const float* a, int64_t n, const float* b, int64_t nb,
                     float eps, float* out, int64_t no) {
  assert(n == nb && n == no);
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] / (b[i] + eps);
}

// ============================
// Minimal extra primitives needed for full forward
// ============================

// embedding_table: (V,H) => (w1=V, w0=H)
// input_ids: (B,S) int32 => (i1=B, i0=S)
// out: (B,S,H) => (o2=B, o1=S, o0=H)
inline void embedding_lookup_bsh(const float* embedding_table, int64_t w1,
                                 int64_t w0, const int32_t* input_ids,
                                 int64_t i1, int64_t i0, float* out, int64_t o2,
                                 int64_t o1, int64_t o0) {
  assert(o2 == i1 && o1 == i0);
  assert(o0 == w0);
  const int64_t B = i1, S = i0, H = w0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      const int32_t tid = input_ids[b * S + s];
      assert(0 <= tid && tid < w1);
      const float* src = embedding_table + static_cast<int64_t>(tid) * H;
      float* dst = out + (b * S + s) * H;
      std::memcpy(dst, src, static_cast<size_t>(H) * sizeof(float));
    }
  }
}

// (B,T,H,D) -> (B,H,T,D) : permute(0,2,1,3)
inline void transpose_bthd_to_bhtd(const float* x, int64_t x3, int64_t x2,
                                   int64_t x1, int64_t x0,  // (B,T,H,D)
                                   float* out, int64_t o3, int64_t o2,
                                   int64_t o1, int64_t o0) {
  // (B,H,T,D)
  assert(o3 == x3 && o2 == x1 && o1 == x2 && o0 == x0);
  const int64_t B = x3, T = x2, H = x1, D = x0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t h = 0; h < H; ++h) {
        const float* src = x + off_4d(b, t, h, 0, x2, x1, x0);
        float* dst = out + off_4d(b, h, t, 0, o2, o1, o0);
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
      }
    }
  }
}

// repeat KV heads: x (B,T,KVH,D) -> out (B,T,H,D), where H = KVH * n_rep
inline void repeat_kv_btkd_to_bthd(const float* x, int64_t x3, int64_t x2,
                                   int64_t x1, int64_t x0,  // (B,T,KVH,D)
                                   int64_t n_rep, float* out, int64_t o3,
                                   int64_t o2, int64_t o1, int64_t o0) {
  // (B,T,H,D)
  assert(n_rep >= 1);
  assert(o3 == x3 && o2 == x2 && o0 == x0);
  assert(o1 == x1 * n_rep);
  const int64_t B = x3, T = x2, KVH = x1, D = x0;
  const int64_t H = KVH * n_rep;

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t kv = 0; kv < KVH; ++kv) {
        const float* src = x + off_4d(b, t, kv, 0, x2, x1, x0);  // (D)
        for (int64_t r = 0; r < n_rep; ++r) {
          const int64_t h = kv * n_rep + r;
          float* dst = out + off_4d(b, t, h, 0, o2, o1, o0);
          std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
        }
      }
    }
  }
}

// KV cache append in-place:
// cache: (B,MAXT,KVH,D)  new_x: (B,S,KVH,D)  write to cache[:, t_off:t_off+S,
// :, :]
inline void kv_cache_append_btkd(float* cache, int64_t c3, int64_t c2,
                                 int64_t c1, int64_t c0,  // (B,MAXT,KVH,D)
                                 const float* new_x, int64_t n3, int64_t n2,
                                 int64_t n1, int64_t n0,  // (B,S,KVH,D)
                                 int64_t t_off) {
  assert(c3 == n3 && c1 == n1 && c0 == n0);
  assert(0 <= t_off && t_off + n2 <= c2);
  const int64_t B = n3, S = n2, KVH = n1, D = n0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t kv = 0; kv < KVH; ++kv) {
        const float* src = new_x + off_4d(b, s, kv, 0, n2, n1, n0);
        float* dst = cache + off_4d(b, t_off + s, kv, 0, c2, c1, c0);
        std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
      }
    }
  }
}

// ============================
// silu_inplace
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ============================
inline void silu_inplace(float* x, int64_t n) {
  assert(x != nullptr);
  for (int64_t i = 0; i < n; ++i) {
    const float v = x[i];
    // sigmoid(v) = 1/(1+exp(-v))
    const float s = 1.0f / (1.0f + std::exp(-v));
    x[i] = v * s;
  }
}

// ============================
// apply_rotary_pos_emb_bshd (self-contained, no off_* helpers)
// q: (B,S,H,D)    => (q3,q2,q1,q0) = (B,S,H,D)
// k: (B,S,KVH,D)  => (k3,k2,k1,k0) = (B,S,KVH,D)
// cos/sin: (S,D)  => (c1,c0) = (S,D), contiguous
//
// In-place update q and k.
//
// Matches your Python logic:
// rotate_half(x) = concat(-x[..., D/2:], x[..., :D/2])
// q' = q*cos + rotate_half(q)*sin
// k' = k*cos + rotate_half(k)*sin
// ============================
inline void apply_rotary_pos_emb_bshd(float* q, int64_t q3, int64_t q2,
                                      int64_t q1, int64_t q0, float* k,
                                      int64_t k3, int64_t k2, int64_t k1,
                                      int64_t k0, const float* cos,
                                      const float* sin, int64_t c1,
                                      int64_t c0) {
  assert(q != nullptr);
  assert(k != nullptr);
  assert(cos != nullptr);
  assert(sin != nullptr);

  // shape checks (negative-index convention)
  assert(q3 == k3);  // B
  assert(q2 == k2);  // S
  assert(q0 == k0);  // D
  assert(c1 == q2);  // S
  assert(c0 == q0);  // D

  const int64_t B = q3;
  const int64_t S = q2;
  const int64_t H = q1;
  const int64_t KVH = k1;
  const int64_t D = q0;

  assert(D > 0);
  assert((D % 2) == 0);
  const int64_t half = D / 2;

  // Offset formulas (row-major, a0 contiguous):
  // q[b,s,h,d] index = (((b*S + s)*H + h)*D + d)
  // k[b,s,h,d] index = (((b*S + s)*KVH + h)*D + d)
  // cos[s,d] index = (s*D + d)
  auto q_index = [S, H, D](int64_t b, int64_t s, int64_t h,
                           int64_t d) -> int64_t {
    return (((b * S + s) * H + h) * D + d);
  };
  auto k_index = [S, KVH, D](int64_t b, int64_t s, int64_t h,
                             int64_t d) -> int64_t {
    return (((b * S + s) * KVH + h) * D + d);
  };
  auto cs_index = [D](int64_t s, int64_t d) -> int64_t { return (s * D + d); };

  // ---- q ----
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      const float* cos_row = cos + cs_index(s, 0);
      const float* sin_row = sin + cs_index(s, 0);

      for (int64_t h = 0; h < H; ++h) {
        const int64_t base = q_index(b, s, h, 0);

        // d in [0, half): pair (d, d+half)
        for (int64_t d = 0; d < half; ++d) {
          const float x1 = q[base + d];
          const float x2 = q[base + d + half];

          const float c1v = cos_row[d];
          const float s1v = sin_row[d];
          const float c2v = cos_row[d + half];
          const float s2v = sin_row[d + half];

          // q[d]        = x1*cos[d] + (-x2)*sin[d]
          // q[d+half]   = x2*cos[d+half] + x1*sin[d+half]
          q[base + d] = x1 * c1v - x2 * s1v;
          q[base + d + half] = x2 * c2v + x1 * s2v;
        }
      }
    }
  }

  // ---- k ----
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      const float* cos_row = cos + cs_index(s, 0);
      const float* sin_row = sin + cs_index(s, 0);

      for (int64_t h = 0; h < KVH; ++h) {
        const int64_t base = k_index(b, s, h, 0);

        for (int64_t d = 0; d < half; ++d) {
          const float x1 = k[base + d];
          const float x2 = k[base + d + half];

          const float c1v = cos_row[d];
          const float s1v = sin_row[d];
          const float c2v = cos_row[d + half];
          const float s2v = sin_row[d + half];

          k[base + d] = x1 * c1v - x2 * s1v;
          k[base + d + half] = x2 * c2v + x1 * s2v;
        }
      }
    }
  }
}
}  // namespace celer_infer

#endif  // CELER_INFER_INCLUDE_CELER_INFER_TENSOR_HPP

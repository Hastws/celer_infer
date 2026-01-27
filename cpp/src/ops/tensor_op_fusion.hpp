// tensor_op_fusion.hpp - Fused tensor operations for improved performance
// ============================================================================
// Fused operators reduce memory bandwidth by combining multiple operations:
//   - fused_rms_norm_linear: RMSNorm + Linear projection
//   - fused_swiglu: SiLU(gate) * up in one pass
//   - fused_attention: Q@K^T -> scale -> mask -> softmax -> @V
//   - fused_ffn: gate + up + silu*up + down in minimal memory passes
// ============================================================================

#ifndef TENSOR_OP_FUSION_HPP
#define TENSOR_OP_FUSION_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cassert>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace fusion {

// ============================================================================
// Fused RMSNorm + Linear: out = Linear(RMSNorm(x))
// Saves one memory round-trip by not materializing normalized intermediate
// ============================================================================
inline void fused_rms_norm_linear(
    const float* x, int64_t batch_seq, int64_t hidden,
    const float* rms_weight, float eps,
    const float* linear_weight, int64_t out_dim,  // weight shape: (out_dim, hidden)
    float* out) {
  
  for (int64_t i = 0; i < batch_seq; ++i) {
    const float* row = x + i * hidden;
    
    // Compute RMS
    float ms = 0.0f;
#ifdef __AVX2__
    __m256 vms = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 8 <= hidden; j += 8) {
      __m256 v = _mm256_loadu_ps(row + j);
      vms = _mm256_fmadd_ps(v, v, vms);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(vms);
    __m128 hi = _mm256_extractf128_ps(vms, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    ms = _mm_cvtss_f32(sums);
    for (; j < hidden; ++j) ms += row[j] * row[j];
#else
    for (int64_t j = 0; j < hidden; ++j) ms += row[j] * row[j];
#endif
    ms /= static_cast<float>(hidden);
    float inv_rms = 1.0f / std::sqrt(ms + eps);
    
    // Fused: normalize and project in one pass
    float* orow = out + i * out_dim;
    for (int64_t o = 0; o < out_dim; ++o) {
      const float* w = linear_weight + o * hidden;
      float acc = 0.0f;
      
#ifdef __AVX2__
      __m256 vacc = _mm256_setzero_ps();
      __m256 vinv = _mm256_set1_ps(inv_rms);
      int64_t k = 0;
      for (; k + 8 <= hidden; k += 8) {
        __m256 vx = _mm256_loadu_ps(row + k);
        __m256 vrms = _mm256_loadu_ps(rms_weight + k);
        __m256 vw = _mm256_loadu_ps(w + k);
        // normalized = x * inv_rms * rms_weight
        __m256 vnorm = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vrms);
        vacc = _mm256_fmadd_ps(vnorm, vw, vacc);
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
      for (; k < hidden; ++k) {
        float norm_val = row[k] * inv_rms * rms_weight[k];
        acc += norm_val * w[k];
      }
#else
      for (int64_t k = 0; k < hidden; ++k) {
        float norm_val = row[k] * inv_rms * rms_weight[k];
        acc += norm_val * w[k];
      }
#endif
      orow[o] = acc;
    }
  }
}

// ============================================================================
// Fused SwiGLU: out = SiLU(gate_proj(x)) * up_proj(x)
// Computes both projections and activation in one memory pass
// ============================================================================
inline void fused_gate_up_swiglu(
    const float* x, int64_t batch_seq, int64_t hidden,
    const float* w_gate, const float* w_up, int64_t inter,  // (inter, hidden)
    float* out) {  // (batch_seq, inter)
  
  for (int64_t i = 0; i < batch_seq; ++i) {
    const float* row = x + i * hidden;
    float* orow = out + i * inter;
    
    for (int64_t o = 0; o < inter; ++o) {
      const float* wg = w_gate + o * hidden;
      const float* wu = w_up + o * hidden;
      
      float gate_val = 0.0f;
      float up_val = 0.0f;
      
#ifdef __AVX2__
      __m256 vgate = _mm256_setzero_ps();
      __m256 vup = _mm256_setzero_ps();
      int64_t k = 0;
      for (; k + 8 <= hidden; k += 8) {
        __m256 vx = _mm256_loadu_ps(row + k);
        __m256 vwg = _mm256_loadu_ps(wg + k);
        __m256 vwu = _mm256_loadu_ps(wu + k);
        vgate = _mm256_fmadd_ps(vx, vwg, vgate);
        vup = _mm256_fmadd_ps(vx, vwu, vup);
      }
      // Horizontal sums
      __m128 lo = _mm256_castps256_ps128(vgate);
      __m128 hi = _mm256_extractf128_ps(vgate, 1);
      lo = _mm_add_ps(lo, hi);
      __m128 shuf = _mm_movehdup_ps(lo);
      __m128 sums = _mm_add_ps(lo, shuf);
      shuf = _mm_movehl_ps(shuf, sums);
      sums = _mm_add_ss(sums, shuf);
      gate_val = _mm_cvtss_f32(sums);
      
      lo = _mm256_castps256_ps128(vup);
      hi = _mm256_extractf128_ps(vup, 1);
      lo = _mm_add_ps(lo, hi);
      shuf = _mm_movehdup_ps(lo);
      sums = _mm_add_ps(lo, shuf);
      shuf = _mm_movehl_ps(shuf, sums);
      sums = _mm_add_ss(sums, shuf);
      up_val = _mm_cvtss_f32(sums);
      
      for (; k < hidden; ++k) {
        gate_val += row[k] * wg[k];
        up_val += row[k] * wu[k];
      }
#else
      for (int64_t k = 0; k < hidden; ++k) {
        gate_val += row[k] * wg[k];
        up_val += row[k] * wu[k];
      }
#endif
      
      // SiLU(gate) * up
      float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
      orow[o] = (gate_val * sigmoid_gate) * up_val;
    }
  }
}

// ============================================================================
// Fused Attention: scores -> scale -> causal_mask -> softmax -> output
// Computes attention without materializing full score matrix
// ============================================================================
inline void fused_attention_forward(
    const float* q, const float* k, const float* v,
    int64_t B, int64_t H, int64_t S, int64_t D,
    float scale, int64_t past_len,
    float* out) {
  
  const int64_t T = past_len + S;  // Total sequence length
  const float neg_inf = -1e9f;
  
  // Temporary buffer for one row of attention scores
  // In production, use thread-local or pre-allocated buffer
  std::vector<float> scores_row(T);
  
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t sq = 0; sq < S; ++sq) {
        const float* q_vec = q + ((b * H + h) * S + sq) * D;
        
        // Compute Q @ K^T for this query position
        float max_score = neg_inf;
        for (int64_t sk = 0; sk < T; ++sk) {
          const float* k_vec = k + ((b * H + h) * T + sk) * D;
          
          float score = 0.0f;
#ifdef __AVX2__
          __m256 vsum = _mm256_setzero_ps();
          int64_t d = 0;
          for (; d + 8 <= D; d += 8) {
            __m256 vq = _mm256_loadu_ps(q_vec + d);
            __m256 vk = _mm256_loadu_ps(k_vec + d);
            vsum = _mm256_fmadd_ps(vq, vk, vsum);
          }
          __m128 lo = _mm256_castps256_ps128(vsum);
          __m128 hi = _mm256_extractf128_ps(vsum, 1);
          lo = _mm_add_ps(lo, hi);
          __m128 shuf = _mm_movehdup_ps(lo);
          __m128 sums = _mm_add_ps(lo, shuf);
          shuf = _mm_movehl_ps(shuf, sums);
          sums = _mm_add_ss(sums, shuf);
          score = _mm_cvtss_f32(sums);
          for (; d < D; ++d) score += q_vec[d] * k_vec[d];
#else
          for (int64_t d = 0; d < D; ++d) score += q_vec[d] * k_vec[d];
#endif
          score *= scale;
          
          // Causal mask: query at position sq+past_len can only attend to positions <= sq+past_len
          if (sk > sq + past_len) {
            score = neg_inf;
          }
          
          scores_row[sk] = score;
          if (score > max_score) max_score = score;
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (int64_t sk = 0; sk < T; ++sk) {
          scores_row[sk] = std::exp(scores_row[sk] - max_score);
          sum_exp += scores_row[sk];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int64_t sk = 0; sk < T; ++sk) {
          scores_row[sk] *= inv_sum;
        }
        
        // Compute weighted sum: out = probs @ V
        float* out_vec = out + ((b * H + h) * S + sq) * D;
        std::memset(out_vec, 0, D * sizeof(float));
        
        for (int64_t sk = 0; sk < T; ++sk) {
          const float prob = scores_row[sk];
          const float* v_vec = v + ((b * H + h) * T + sk) * D;
          
#ifdef __AVX2__
          __m256 vp = _mm256_set1_ps(prob);
          int64_t d = 0;
          for (; d + 8 <= D; d += 8) {
            __m256 vo = _mm256_loadu_ps(out_vec + d);
            __m256 vv = _mm256_loadu_ps(v_vec + d);
            vo = _mm256_fmadd_ps(vp, vv, vo);
            _mm256_storeu_ps(out_vec + d, vo);
          }
          for (; d < D; ++d) out_vec[d] += prob * v_vec[d];
#else
          for (int64_t d = 0; d < D; ++d) out_vec[d] += prob * v_vec[d];
#endif
        }
      }
    }
  }
}

// ============================================================================
// Fused complete FFN block: RMSNorm -> Gate+Up -> SwiGLU -> Down -> Residual
// ============================================================================
inline void fused_ffn_block(
    const float* x, int64_t batch_seq, int64_t hidden,
    const float* rms_weight, float eps,
    const float* w_gate, const float* w_up, const float* w_down, int64_t inter,
    float* out,       // Output (includes residual)
    float* temp_mid)  // Temporary buffer (batch_seq, inter)
{
  // Step 1: Fused RMSNorm + Gate/Up projection + SwiGLU
  for (int64_t i = 0; i < batch_seq; ++i) {
    const float* row = x + i * hidden;
    
    // Compute RMS
    float ms = 0.0f;
    for (int64_t j = 0; j < hidden; ++j) ms += row[j] * row[j];
    ms /= static_cast<float>(hidden);
    float inv_rms = 1.0f / std::sqrt(ms + eps);
    
    // Fused gate/up projection with SwiGLU
    float* mid_row = temp_mid + i * inter;
    for (int64_t o = 0; o < inter; ++o) {
      const float* wg = w_gate + o * hidden;
      const float* wu = w_up + o * hidden;
      
      float gate_val = 0.0f;
      float up_val = 0.0f;
      for (int64_t k = 0; k < hidden; ++k) {
        float norm_val = row[k] * inv_rms * rms_weight[k];
        gate_val += norm_val * wg[k];
        up_val += norm_val * wu[k];
      }
      
      // SiLU(gate) * up
      float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
      mid_row[o] = (gate_val * sigmoid_gate) * up_val;
    }
  }
  
  // Step 2: Down projection + residual
  for (int64_t i = 0; i < batch_seq; ++i) {
    const float* mid_row = temp_mid + i * inter;
    float* out_row = out + i * hidden;
    const float* x_row = x + i * hidden;
    
    for (int64_t o = 0; o < hidden; ++o) {
      const float* wd = w_down + o * inter;
      float acc = 0.0f;
      
#ifdef __AVX2__
      __m256 vacc = _mm256_setzero_ps();
      int64_t k = 0;
      for (; k + 8 <= inter; k += 8) {
        __m256 vm = _mm256_loadu_ps(mid_row + k);
        __m256 vw = _mm256_loadu_ps(wd + k);
        vacc = _mm256_fmadd_ps(vm, vw, vacc);
      }
      __m128 lo = _mm256_castps256_ps128(vacc);
      __m128 hi = _mm256_extractf128_ps(vacc, 1);
      lo = _mm_add_ps(lo, hi);
      __m128 shuf = _mm_movehdup_ps(lo);
      __m128 sums = _mm_add_ps(lo, shuf);
      shuf = _mm_movehl_ps(shuf, sums);
      sums = _mm_add_ss(sums, shuf);
      acc = _mm_cvtss_f32(sums);
      for (; k < inter; ++k) acc += mid_row[k] * wd[k];
#else
      for (int64_t k = 0; k < inter; ++k) acc += mid_row[k] * wd[k];
#endif
      
      // Residual
      out_row[o] = x_row[o] + acc;
    }
  }
}

}  // namespace fusion

#endif  // TENSOR_OP_FUSION_HPP

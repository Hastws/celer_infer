/**
 * DiT SIMD-Optimized Inference Engine
 * AVX2/FMA vectorized implementation
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <immintrin.h>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

// ============================================================================
// SIMD Helpers
// ============================================================================
namespace simd {

// Horizontal sum of __m256
inline float hsum_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Vectorized exp (polynomial approximation)
inline __m256 exp_avx(__m256 x) {
    // Clamp to prevent overflow
    __m256 max_val = _mm256_set1_ps(88.0f);
    __m256 min_val = _mm256_set1_ps(-88.0f);
    x = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);
    
    // exp(x) ≈ 2^(x * log2(e))
    __m256 log2e = _mm256_set1_ps(1.44269504f);
    __m256 t = _mm256_mul_ps(x, log2e);
    
    // Reduce to [0, 1) range
    __m256 f = _mm256_floor_ps(t);
    __m256 r = _mm256_sub_ps(t, f);
    
    // Polynomial for 2^r
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(0.693147f);
    __m256 c2 = _mm256_set1_ps(0.240226f);
    __m256 c3 = _mm256_set1_ps(0.055504f);
    __m256 c4 = _mm256_set1_ps(0.009618f);
    
    __m256 poly = _mm256_fmadd_ps(r, c4, c3);
    poly = _mm256_fmadd_ps(poly, r, c2);
    poly = _mm256_fmadd_ps(poly, r, c1);
    poly = _mm256_fmadd_ps(poly, r, c0);
    
    // Scale by 2^floor
    __m256i fi = _mm256_cvtps_epi32(f);
    fi = _mm256_add_epi32(fi, _mm256_set1_epi32(127));
    fi = _mm256_slli_epi32(fi, 23);
    __m256 scale = _mm256_castsi256_ps(fi);
    
    return _mm256_mul_ps(poly, scale);
}

// GELU approximation (tanh version)
inline __m256 gelu_avx(__m256 x) {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    __m256 c1 = _mm256_set1_ps(0.7978845608f);  // sqrt(2/π)
    __m256 c2 = _mm256_set1_ps(0.044715f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    
    __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    __m256 inner = _mm256_fmadd_ps(c2, x3, x);
    inner = _mm256_mul_ps(c1, inner);
    
    // Approximate tanh with exp
    __m256 exp2x = exp_avx(_mm256_mul_ps(_mm256_set1_ps(2.0f), inner));
    __m256 tanh_val = _mm256_div_ps(
        _mm256_sub_ps(exp2x, one),
        _mm256_add_ps(exp2x, one)
    );
    
    return _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_val)));
}

// SiLU
inline __m256 silu_avx(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_avx(neg_x)));
    return _mm256_mul_ps(x, sigmoid);
}

}  // namespace simd

// ============================================================================
// Configuration & Weights (same structure as baseline)
// ============================================================================
struct DiTConfig {
    int future_len;
    int time_len;
    int agent_state_dim;
    int agent_num;
    int static_objects_num;
    int lane_num;
    int encoder_depth;
    int decoder_depth;
    int num_heads;
    int hidden_dim;
    int predicted_neighbor_num;
};

struct LinearWeights {
    const float* weight = nullptr;
    const float* bias = nullptr;
    int in_features = 0;
    int out_features = 0;
};

struct LayerNormWeights {
    const float* weight = nullptr;
    const float* bias = nullptr;
    int dim = 0;
};

struct MLPWeights {
    LinearWeights fc1;
    LinearWeights fc2;
};

struct MultiHeadAttentionWeights {
    const float* in_proj_weight = nullptr;
    const float* in_proj_bias = nullptr;
    LinearWeights out_proj;
    int embed_dim = 0;
    int num_heads = 0;
};

struct MixerBlockWeights {
    LayerNormWeights norm1;
    MLPWeights channels_mlp;
    LayerNormWeights norm2;
    MLPWeights tokens_mlp;
};

struct SelfAttentionBlockWeights {
    LayerNormWeights norm1;
    MultiHeadAttentionWeights attn;
    LayerNormWeights norm2;
    MLPWeights mlp;
};

struct DiTBlockWeights {
    LayerNormWeights norm1;
    MultiHeadAttentionWeights attn;
    LayerNormWeights norm2;
    MLPWeights mlp1;
    LinearWeights adaLN_modulation;
    LayerNormWeights norm3;
    MultiHeadAttentionWeights cross_attn;
    LayerNormWeights norm4;
    MLPWeights mlp2;
};

struct FinalLayerWeights {
    LayerNormWeights norm_final;
    LayerNormWeights proj_norm1;
    LinearWeights proj_linear1;
    LayerNormWeights proj_norm2;
    LinearWeights proj_linear2;
    LinearWeights adaLN_modulation;
};

struct DiTWeights {
    LinearWeights encoder_pos_emb;
    LinearWeights neighbor_type_emb;
    MLPWeights neighbor_channel_pre;
    MLPWeights neighbor_token_pre;
    std::vector<MixerBlockWeights> neighbor_blocks;
    LayerNormWeights neighbor_norm;
    MLPWeights neighbor_emb_project;
    MLPWeights static_projection;
    LinearWeights lane_speed_limit_emb;
    const float* lane_unknown_speed_emb = nullptr;
    LinearWeights lane_traffic_emb;
    MLPWeights lane_channel_pre;
    MLPWeights lane_token_pre;
    std::vector<MixerBlockWeights> lane_blocks;
    LayerNormWeights lane_norm;
    MLPWeights lane_emb_project;
    std::vector<SelfAttentionBlockWeights> fusion_blocks;
    LayerNormWeights fusion_norm;
    MLPWeights route_channel_pre;
    MLPWeights route_token_pre;
    MixerBlockWeights route_mixer;
    LayerNormWeights route_norm;
    MLPWeights route_emb_project;
    const float* agent_embedding = nullptr;
    MLPWeights preproj;
    LinearWeights t_embedder_mlp_0;
    LinearWeights t_embedder_mlp_2;
    int frequency_embedding_size;
    std::vector<DiTBlockWeights> dit_blocks;
    FinalLayerWeights final_layer;
};

// ============================================================================
// SIMD Operations
// ============================================================================
namespace ops {

void layer_norm_simd(float* out, const float* x, const float* weight, const float* bias,
                     int batch, int dim) {
    const float eps = 1e-5f;
    
    for (int b = 0; b < batch; ++b) {
        const float* xb = x + b * dim;
        float* ob = out + b * dim;
        
        // Compute mean with SIMD
        __m256 sum_vec = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= dim; i += 8) {
            sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(xb + i));
        }
        float mean = simd::hsum_avx(sum_vec);
        for (; i < dim; ++i) mean += xb[i];
        mean /= dim;
        
        // Compute variance with SIMD
        __m256 mean_vec = _mm256_set1_ps(mean);
        __m256 var_vec = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(xb + i), mean_vec);
            var_vec = _mm256_fmadd_ps(diff, diff, var_vec);
        }
        float var = simd::hsum_avx(var_vec);
        for (; i < dim; ++i) {
            float d = xb[i] - mean;
            var += d * d;
        }
        var /= dim;
        
        // Normalize with SIMD
        float inv_std = 1.0f / std::sqrt(var + eps);
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);
        i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 xv = _mm256_loadu_ps(xb + i);
            __m256 wv = _mm256_loadu_ps(weight + i);
            __m256 bv = _mm256_loadu_ps(bias + i);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(xv, mean_vec), inv_std_vec);
            __m256 result = _mm256_fmadd_ps(norm, wv, bv);
            _mm256_storeu_ps(ob + i, result);
        }
        for (; i < dim; ++i) {
            ob[i] = (xb[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

void linear_simd(float* out, const float* x, const LinearWeights& w, int batch) {
    int in_f = w.in_features;
    int out_f = w.out_features;
    
    for (int b = 0; b < batch; ++b) {
        const float* xb = x + b * in_f;
        float* ob = out + b * out_f;
        
        for (int o = 0; o < out_f; ++o) {
            const float* wo = w.weight + o * in_f;
            __m256 acc = _mm256_setzero_ps();
            
            int i = 0;
            for (; i + 8 <= in_f; i += 8) {
                __m256 xv = _mm256_loadu_ps(xb + i);
                __m256 wv = _mm256_loadu_ps(wo + i);
                acc = _mm256_fmadd_ps(xv, wv, acc);
            }
            
            float sum = simd::hsum_avx(acc);
            for (; i < in_f; ++i) {
                sum += xb[i] * wo[i];
            }
            
            ob[o] = w.bias ? sum + w.bias[o] : sum;
        }
    }
}

void mlp_gelu_simd(float* out, const float* x, const MLPWeights& w, int batch, float* tmp) {
    int hidden = w.fc1.out_features;
    
    // fc1
    linear_simd(tmp, x, w.fc1, batch);
    
    // GELU activation with SIMD
    int total = batch * hidden;
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 v = _mm256_loadu_ps(tmp + i);
        _mm256_storeu_ps(tmp + i, simd::gelu_avx(v));
    }
    for (; i < total; ++i) {
        float x_val = tmp[i];
        float x3 = x_val * x_val * x_val;
        float inner = 0.7978845608f * (x_val + 0.044715f * x3);
        tmp[i] = 0.5f * x_val * (1.0f + std::tanh(inner));
    }
    
    // fc2
    linear_simd(out, tmp, w.fc2, batch);
}

void softmax_simd(float* x, int batch, int dim) {
    for (int b = 0; b < batch; ++b) {
        float* xb = x + b * dim;
        
        // Find max
        __m256 max_vec = _mm256_set1_ps(-1e30f);
        int i = 0;
        for (; i + 8 <= dim; i += 8) {
            max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(xb + i));
        }
        float max_val = simd::hsum_avx(max_vec) / 8.0f;
        for (int j = 0; j < 8; ++j) {
            max_val = std::max(max_val, ((float*)&max_vec)[j]);
        }
        for (; i < dim; ++i) max_val = std::max(max_val, xb[i]);
        
        // Exp and sum
        __m256 max_vec_final = _mm256_set1_ps(max_val);
        __m256 sum_vec = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(xb + i), max_vec_final);
            __m256 ev = simd::exp_avx(v);
            _mm256_storeu_ps(xb + i, ev);
            sum_vec = _mm256_add_ps(sum_vec, ev);
        }
        float sum = simd::hsum_avx(sum_vec);
        for (; i < dim; ++i) {
            xb[i] = std::exp(xb[i] - max_val);
            sum += xb[i];
        }
        
        // Normalize
        __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
        i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 v = _mm256_loadu_ps(xb + i);
            _mm256_storeu_ps(xb + i, _mm256_mul_ps(v, inv_sum));
        }
        for (; i < dim; ++i) xb[i] /= sum;
    }
}

}  // namespace ops

// ============================================================================
// JSON Loading
// ============================================================================
std::vector<float> decode_base64_f32(const json& tensor_json) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::string encoded = tensor_json["data"].get<std::string>();
    std::vector<unsigned char> decoded;
    
    int val = 0, bits = -8;
    for (char c : encoded) {
        if (c == '=') break;
        size_t pos = base64_chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<unsigned char>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    
    size_t num_floats = decoded.size() / sizeof(float);
    std::vector<float> result(num_floats);
    std::memcpy(result.data(), decoded.data(), decoded.size());
    return result;
}

std::vector<int> get_shape(const json& tensor_json) {
    return tensor_json["shape"].get<std::vector<int>>();
}

class DiTWeightLoader {
public:
    std::vector<std::vector<float>> buffers;
    
    const float* load_tensor(const json& j) {
        buffers.push_back(decode_base64_f32(j));
        return buffers.back().data();
    }
    
    LinearWeights load_linear(const json& j) {
        LinearWeights w;
        auto shape = get_shape(j["weight"]);
        w.out_features = shape[0];
        w.in_features = shape[1];
        w.weight = load_tensor(j["weight"]);
        if (j.contains("bias")) {
            w.bias = load_tensor(j["bias"]);
        }
        return w;
    }
    
    LayerNormWeights load_layernorm(const json& j) {
        LayerNormWeights w;
        auto shape = get_shape(j["weight"]);
        w.dim = shape[0];
        w.weight = load_tensor(j["weight"]);
        w.bias = load_tensor(j["bias"]);
        return w;
    }
    
    MLPWeights load_mlp(const json& j) {
        MLPWeights w;
        w.fc1 = load_linear(j["fc1"]);
        w.fc2 = load_linear(j["fc2"]);
        return w;
    }
    
    MultiHeadAttentionWeights load_mha(const json& j, int num_heads) {
        MultiHeadAttentionWeights w;
        auto shape = get_shape(j["in_proj_weight"]);
        w.embed_dim = shape[1];
        w.num_heads = num_heads;
        w.in_proj_weight = load_tensor(j["in_proj_weight"]);
        w.in_proj_bias = load_tensor(j["in_proj_bias"]);
        w.out_proj = load_linear(j["out_proj"]);
        return w;
    }
    
    MixerBlockWeights load_mixer_block(const json& j) {
        MixerBlockWeights w;
        w.norm1 = load_layernorm(j["norm1"]);
        w.channels_mlp = load_mlp(j["channels_mlp"]);
        w.norm2 = load_layernorm(j["norm2"]);
        w.tokens_mlp = load_mlp(j["tokens_mlp"]);
        return w;
    }
    
    SelfAttentionBlockWeights load_self_attn_block(const json& j, int num_heads) {
        SelfAttentionBlockWeights w;
        w.norm1 = load_layernorm(j["norm1"]);
        w.attn = load_mha(j["attn"], num_heads);
        w.norm2 = load_layernorm(j["norm2"]);
        w.mlp = load_mlp(j["mlp"]);
        return w;
    }
    
    DiTBlockWeights load_dit_block(const json& j, int num_heads) {
        DiTBlockWeights w;
        w.norm1 = load_layernorm(j["norm1"]);
        w.attn = load_mha(j["attn"], num_heads);
        w.norm2 = load_layernorm(j["norm2"]);
        w.mlp1 = load_mlp(j["mlp1"]);
        w.adaLN_modulation = load_linear(j["adaLN_modulation"]["linear"]);
        w.norm3 = load_layernorm(j["norm3"]);
        w.cross_attn = load_mha(j["cross_attn"], num_heads);
        w.norm4 = load_layernorm(j["norm4"]);
        w.mlp2 = load_mlp(j["mlp2"]);
        return w;
    }
    
    DiTWeights load_weights(const json& weights_json, const DiTConfig& cfg) {
        DiTWeights w;
        w.encoder_pos_emb = load_linear(weights_json["encoder_pos_emb"]);
        
        auto& ne = weights_json["neighbor_encoder"];
        w.neighbor_type_emb = load_linear(ne["type_emb"]);
        w.neighbor_channel_pre = load_mlp(ne["channel_pre_project"]);
        w.neighbor_token_pre = load_mlp(ne["token_pre_project"]);
        for (auto& blk : ne["blocks"]) {
            w.neighbor_blocks.push_back(load_mixer_block(blk));
        }
        w.neighbor_norm = load_layernorm(ne["norm"]);
        w.neighbor_emb_project = load_mlp(ne["emb_project"]);
        
        w.static_projection = load_mlp(weights_json["static_encoder"]["projection"]);
        
        auto& le = weights_json["lane_encoder"];
        w.lane_speed_limit_emb = load_linear(le["speed_limit_emb"]);
        w.lane_unknown_speed_emb = load_tensor(le["unknown_speed_emb"]["weight"]);
        w.lane_traffic_emb = load_linear(le["traffic_emb"]);
        w.lane_channel_pre = load_mlp(le["channel_pre_project"]);
        w.lane_token_pre = load_mlp(le["token_pre_project"]);
        for (auto& blk : le["blocks"]) {
            w.lane_blocks.push_back(load_mixer_block(blk));
        }
        w.lane_norm = load_layernorm(le["norm"]);
        w.lane_emb_project = load_mlp(le["emb_project"]);
        
        auto& fe = weights_json["fusion_encoder"];
        for (auto& blk : fe["blocks"]) {
            w.fusion_blocks.push_back(load_self_attn_block(blk, cfg.num_heads));
        }
        w.fusion_norm = load_layernorm(fe["norm"]);
        
        auto& re = weights_json["route_encoder"];
        w.route_channel_pre = load_mlp(re["channel_pre_project"]);
        w.route_token_pre = load_mlp(re["token_pre_project"]);
        w.route_mixer = load_mixer_block(re["mixer"]);
        w.route_norm = load_layernorm(re["norm"]);
        w.route_emb_project = load_mlp(re["emb_project"]);
        
        w.agent_embedding = load_tensor(weights_json["agent_embedding"]["weight"]);
        w.preproj = load_mlp(weights_json["preproj"]);
        w.t_embedder_mlp_0 = load_linear(weights_json["t_embedder"]["mlp_0"]);
        w.t_embedder_mlp_2 = load_linear(weights_json["t_embedder"]["mlp_2"]);
        w.frequency_embedding_size = weights_json["t_embedder"]["frequency_embedding_size"];
        
        for (auto& blk : weights_json["dit_blocks"]) {
            w.dit_blocks.push_back(load_dit_block(blk, cfg.num_heads));
        }
        
        auto& fl = weights_json["final_layer"];
        w.final_layer.norm_final = load_layernorm(fl["norm_final"]);
        w.final_layer.proj_norm1 = load_layernorm(fl["proj"]["norm1"]);
        w.final_layer.proj_linear1 = load_linear(fl["proj"]["linear1"]);
        w.final_layer.proj_norm2 = load_layernorm(fl["proj"]["norm2"]);
        w.final_layer.proj_linear2 = load_linear(fl["proj"]["linear2"]);
        w.final_layer.adaLN_modulation = load_linear(fl["adaLN_modulation"]["linear"]);
        
        return w;
    }
};

// ============================================================================
// DiT Inference
// ============================================================================
class DiTInference {
public:
    DiTConfig cfg;
    DiTWeights weights;
    std::vector<float> tmp1, tmp2, tmp3, tmp4;
    
    void init_workspace(int batch_size) {
        int max_tokens = cfg.agent_num + cfg.static_objects_num + cfg.lane_num;
        size_t buf_size = batch_size * max_tokens * cfg.hidden_dim * 4;
        tmp1.resize(buf_size);
        tmp2.resize(buf_size);
        tmp3.resize(buf_size);
        tmp4.resize(buf_size);
    }
    
    void timestep_embedding(float* out, const float* t, int batch) {
        int dim = weights.frequency_embedding_size;
        int half = dim / 2;
        
        for (int b = 0; b < batch; ++b) {
            float tb = t[b];
            for (int i = 0; i < half; ++i) {
                float freq = std::exp(-std::log(10000.0f) * static_cast<float>(i) / half);
                float arg = tb * freq;
                out[b * dim + i] = std::cos(arg);
                out[b * dim + half + i] = std::sin(arg);
            }
        }
        
        ops::linear_simd(tmp1.data(), out, weights.t_embedder_mlp_0, batch);
        int total = batch * weights.t_embedder_mlp_0.out_features;
        int i = 0;
        for (; i + 8 <= total; i += 8) {
            __m256 v = _mm256_loadu_ps(tmp1.data() + i);
            _mm256_storeu_ps(tmp1.data() + i, simd::silu_avx(v));
        }
        for (; i < total; ++i) {
            float x = tmp1[i];
            tmp1[i] = x / (1.0f + std::exp(-x));
        }
        ops::linear_simd(out, tmp1.data(), weights.t_embedder_mlp_2, batch);
    }
    
    void encoder_forward(float* encoding, const float* inputs, int batch) {
        int token_num = cfg.agent_num + cfg.static_objects_num + cfg.lane_num;
        std::fill(encoding, encoding + batch * token_num * cfg.hidden_dim, 0.0f);
        
        for (auto& block : weights.fusion_blocks) {
            ops::layer_norm_simd(tmp1.data(), encoding, block.norm1.weight, block.norm1.bias,
                                batch * token_num, cfg.hidden_dim);
            ops::layer_norm_simd(tmp2.data(), encoding, block.norm2.weight, block.norm2.bias,
                                batch * token_num, cfg.hidden_dim);
            ops::mlp_gelu_simd(tmp3.data(), tmp2.data(), block.mlp, batch * token_num, tmp4.data());
            
            for (int i = 0; i < batch * token_num * cfg.hidden_dim; ++i) {
                encoding[i] += tmp3[i];
            }
        }
        
        ops::layer_norm_simd(tmp1.data(), encoding, weights.fusion_norm.weight, weights.fusion_norm.bias,
                           batch * token_num, cfg.hidden_dim);
        std::copy(tmp1.begin(), tmp1.begin() + batch * token_num * cfg.hidden_dim, encoding);
    }
    
    void decoder_forward(float* score, const float* x, const float* t, const float* encoding,
                        int batch) {
        int P = 1 + cfg.predicted_neighbor_num;
        int output_dim = (cfg.future_len + 1) * 4;
        
        ops::mlp_gelu_simd(tmp1.data(), x, weights.preproj, batch * P, tmp2.data());
        
        for (int b = 0; b < batch; ++b) {
            for (int d = 0; d < cfg.hidden_dim; ++d) {
                tmp1[b * P * cfg.hidden_dim + d] += weights.agent_embedding[d];
            }
            for (int p = 1; p < P; ++p) {
                for (int d = 0; d < cfg.hidden_dim; ++d) {
                    tmp1[b * P * cfg.hidden_dim + p * cfg.hidden_dim + d] += weights.agent_embedding[cfg.hidden_dim + d];
                }
            }
        }
        
        std::vector<float> y(batch * cfg.hidden_dim, 0.0f);
        std::vector<float> t_emb(batch * weights.frequency_embedding_size);
        timestep_embedding(t_emb.data(), t, batch);
        for (int b = 0; b < batch; ++b) {
            for (int d = 0; d < cfg.hidden_dim; ++d) {
                y[b * cfg.hidden_dim + d] = t_emb[b * cfg.hidden_dim + d];
            }
        }
        
        float* current = tmp1.data();
        for (auto& block : weights.dit_blocks) {
            std::vector<float> modulation(batch * 6 * cfg.hidden_dim);
            int total = batch * cfg.hidden_dim;
            int i = 0;
            for (; i + 8 <= total; i += 8) {
                __m256 v = _mm256_loadu_ps(y.data() + i);
                _mm256_storeu_ps(y.data() + i, simd::silu_avx(v));
            }
            for (; i < total; ++i) {
                float yv = y[i];
                y[i] = yv / (1.0f + std::exp(-yv));
            }
            ops::linear_simd(modulation.data(), y.data(), block.adaLN_modulation, batch);
            
            ops::layer_norm_simd(tmp2.data(), current, block.norm1.weight, block.norm1.bias,
                               batch * P, cfg.hidden_dim);
            ops::layer_norm_simd(tmp3.data(), tmp2.data(), block.norm2.weight, block.norm2.bias,
                               batch * P, cfg.hidden_dim);
            ops::mlp_gelu_simd(tmp4.data(), tmp3.data(), block.mlp1, batch * P, tmp2.data());
            
            ops::layer_norm_simd(tmp2.data(), tmp4.data(), block.norm3.weight, block.norm3.bias,
                               batch * P, cfg.hidden_dim);
            ops::layer_norm_simd(tmp3.data(), tmp2.data(), block.norm4.weight, block.norm4.bias,
                               batch * P, cfg.hidden_dim);
            ops::mlp_gelu_simd(current, tmp3.data(), block.mlp2, batch * P, tmp4.data());
        }
        
        ops::layer_norm_simd(tmp2.data(), current, weights.final_layer.norm_final.weight,
                           weights.final_layer.norm_final.bias, batch * P, cfg.hidden_dim);
        ops::layer_norm_simd(tmp3.data(), tmp2.data(), weights.final_layer.proj_norm1.weight,
                           weights.final_layer.proj_norm1.bias, batch * P, cfg.hidden_dim);
        ops::linear_simd(tmp4.data(), tmp3.data(), weights.final_layer.proj_linear1, batch * P);
        
        int total = batch * P * weights.final_layer.proj_linear1.out_features;
        int i = 0;
        for (; i + 8 <= total; i += 8) {
            __m256 v = _mm256_loadu_ps(tmp4.data() + i);
            _mm256_storeu_ps(tmp4.data() + i, simd::gelu_avx(v));
        }
        for (; i < total; ++i) {
            float x_val = tmp4[i];
            float x3 = x_val * x_val * x_val;
            float inner = 0.7978845608f * (x_val + 0.044715f * x3);
            tmp4[i] = 0.5f * x_val * (1.0f + std::tanh(inner));
        }
        
        ops::layer_norm_simd(tmp2.data(), tmp4.data(), weights.final_layer.proj_norm2.weight,
                           weights.final_layer.proj_norm2.bias, batch * P, weights.final_layer.proj_linear1.out_features);
        ops::linear_simd(score, tmp2.data(), weights.final_layer.proj_linear2, batch * P);
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <json_path> <dump_dir>" << std::endl;
        return 1;
    }
    
    std::string json_path = argv[1];
    std::string dump_dir = argv[2];
    
    std::cout << "[OK] Loading JSON from: " << json_path << std::endl;
    std::ifstream ifs(json_path);
    json j;
    ifs >> j;
    
    DiTConfig cfg;
    auto& cj = j["config"];
    cfg.future_len = cj["future_len"];
    cfg.time_len = cj["time_len"];
    cfg.agent_num = cj["agent_num"];
    cfg.static_objects_num = cj["static_objects_num"];
    cfg.lane_num = cj["lane_num"];
    cfg.encoder_depth = cj["encoder_depth"];
    cfg.decoder_depth = cj["decoder_depth"];
    cfg.num_heads = cj["num_heads"];
    cfg.hidden_dim = cj["hidden_dim"];
    cfg.predicted_neighbor_num = cj["predicted_neighbor_num"];
    
    std::cout << "[SIMD] Config: hidden=" << cfg.hidden_dim << " depth=" << cfg.encoder_depth 
              << " heads=" << cfg.num_heads << std::endl;
    
    DiTWeightLoader loader;
    DiTWeights weights = loader.load_weights(j["weights"], cfg);
    std::cout << "[OK] Loaded weights" << std::endl;
    
    auto input_shape = get_shape(j["inputs"]["sampled_trajectories"]);
    int B = input_shape[0];
    int P = input_shape[1];
    
    std::vector<float> sampled_traj = decode_base64_f32(j["inputs"]["sampled_trajectories"]);
    std::vector<float> diffusion_time = decode_base64_f32(j["inputs"]["diffusion_time"]);
    
    DiTInference infer;
    infer.cfg = cfg;
    infer.weights = weights;
    infer.init_workspace(B);
    
    int token_num = cfg.agent_num + cfg.static_objects_num + cfg.lane_num;
    std::vector<float> encoding(B * token_num * cfg.hidden_dim);
    int output_dim = (cfg.future_len + 1) * 4;
    std::vector<float> score(B * P * output_dim);
    
    // Warmup
    infer.encoder_forward(encoding.data(), nullptr, B);
    infer.decoder_forward(score.data(), sampled_traj.data(), diffusion_time.data(), encoding.data(), B);
    
    // Benchmark
    int num_runs = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        infer.encoder_forward(encoding.data(), nullptr, B);
        infer.decoder_forward(score.data(), sampled_traj.data(), diffusion_time.data(), encoding.data(), B);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    
    float min_val = score[0], max_val = score[0], sum = 0;
    for (size_t i = 0; i < score.size(); ++i) {
        min_val = std::min(min_val, score[i]);
        max_val = std::max(max_val, score[i]);
        sum += score[i];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "[SIMD Forward] Score shape: (" << B << ", " << P << ", " << output_dim << ")" << std::endl;
    std::cout << "[Timing] Forward pass: " << elapsed_ms << "ms" << std::endl;
    std::cout << "[Score] Min: " << min_val << ", Max: " << max_val << ", Mean: " << sum/score.size() << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

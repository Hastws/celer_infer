// minimind_unified.cpp - Unified MiniMind inference with modular backends
// ============================================================================
// This is a single binary that supports all backend combinations:
//   - baseline: Pure C++ implementation
//   - simd: AVX2 vectorized operations  
//   - fusion: Operator fusion for reduced memory bandwidth
//   - quant: INT8 quantization for smaller memory footprint
//   - Any combination: simd+fusion, simd+quant, simd+fusion+quant, etc.
//
// Usage:
//   ./minimind_unified <json_path> <dump_dir> [backend]
//   backend examples: "baseline", "simd", "simd+fusion", "all", etc.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "../ops/tensor_op.hpp"
#include "../ops/backend_config.hpp"

#ifdef __AVX2__
#include "../ops/tensor_op_simd.hpp"
#endif

#include "../ops/tensor_op_fusion.hpp"
#include "../ops/tensor_op_quant.hpp"

#include "../../third_party/nlohmann/json.hpp"

using json = nlohmann::json;
using namespace celer_infer;

// ============================================================================
// Model structs (same as baseline)
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
  bool use_moe = false;
};

struct minimind_layer_weights {
  const float* w_q = nullptr;
  const float* w_k = nullptr;
  const float* w_v = nullptr;
  const float* w_o = nullptr;
  const float* w_gate = nullptr;
  const float* w_up = nullptr;
  const float* w_down = nullptr;
  const float* rms_attn = nullptr;
  const float* rms_ffn = nullptr;
  
  // Quantized weights (populated on demand)
  mutable std::vector<int8_t> w_q_quant;
  mutable std::vector<int8_t> w_k_quant;
  mutable std::vector<int8_t> w_v_quant;
  mutable std::vector<int8_t> w_o_quant;
  mutable std::vector<int8_t> w_gate_quant;
  mutable std::vector<int8_t> w_up_quant;
  mutable std::vector<int8_t> w_down_quant;
  
  mutable quant::PerChannelQuantParams w_q_params;
  mutable quant::PerChannelQuantParams w_k_params;
  mutable quant::PerChannelQuantParams w_v_params;
  mutable quant::PerChannelQuantParams w_o_params;
  mutable quant::PerChannelQuantParams w_gate_params;
  mutable quant::PerChannelQuantParams w_up_params;
  mutable quant::PerChannelQuantParams w_down_params;
  
  mutable bool quantized = false;
};

struct minimind_weights {
  const float* tok_embedding = nullptr;
  const float* final_rms = nullptr;
  const float* lm_head = nullptr;
  const float* rope_cos = nullptr;
  const float* rope_sin = nullptr;
  minimind_layer_weights* layers = nullptr;
  
  // Quantized lm_head
  mutable std::vector<int8_t> lm_head_quant;
  mutable quant::PerChannelQuantParams lm_head_params;
};

struct kv_cache {
  float** k = nullptr;
  float** v = nullptr;
  int64_t max_t = 0;
  int64_t cur_t = 0;
};

struct minimind_workspace {
  float* h0 = nullptr;
  float* h1 = nullptr;
  float* q = nullptr;
  float* k = nullptr;
  float* v = nullptr;
  float* k_rep = nullptr;
  float* v_rep = nullptr;
  float* q_bhsd = nullptr;
  float* k_bhtd = nullptr;
  float* v_bhtd = nullptr;
  float* scores = nullptr;
  float* probs = nullptr;
  float* attn_out = nullptr;
  float* attn_out_bshd = nullptr;
  float* attn_out_flat = nullptr;
  float* ffn_gate = nullptr;
  float* ffn_up = nullptr;
  float* ffn_mid = nullptr;
  float* ffn_out = nullptr;
  float* logits = nullptr;
};

// ============================================================================
// Quantization helper: quantize all weights in a layer
// ============================================================================
void quantize_layer_weights(const minimind_config& cfg, minimind_layer_weights& lw) {
  if (lw.quantized) return;
  
  const int64_t H = cfg.n_heads;
  const int64_t KVH = cfg.n_kv_heads;
  const int64_t D = cfg.head_dim;
  const int64_t hidden = cfg.hidden;
  const int64_t inter = cfg.inter;
  
  // Q: (H*D, hidden)
  lw.w_q_quant.resize(H * D * hidden);
  lw.w_q_params = quant::compute_per_channel_quant_params(lw.w_q, H * D, hidden);
  quant::quantize_weight_per_channel(lw.w_q, H * D, hidden, lw.w_q_params, lw.w_q_quant.data());
  
  // K: (KVH*D, hidden)
  lw.w_k_quant.resize(KVH * D * hidden);
  lw.w_k_params = quant::compute_per_channel_quant_params(lw.w_k, KVH * D, hidden);
  quant::quantize_weight_per_channel(lw.w_k, KVH * D, hidden, lw.w_k_params, lw.w_k_quant.data());
  
  // V: (KVH*D, hidden)
  lw.w_v_quant.resize(KVH * D * hidden);
  lw.w_v_params = quant::compute_per_channel_quant_params(lw.w_v, KVH * D, hidden);
  quant::quantize_weight_per_channel(lw.w_v, KVH * D, hidden, lw.w_v_params, lw.w_v_quant.data());
  
  // O: (hidden, H*D)
  lw.w_o_quant.resize(hidden * H * D);
  lw.w_o_params = quant::compute_per_channel_quant_params(lw.w_o, hidden, H * D);
  quant::quantize_weight_per_channel(lw.w_o, hidden, H * D, lw.w_o_params, lw.w_o_quant.data());
  
  // Gate: (inter, hidden)
  lw.w_gate_quant.resize(inter * hidden);
  lw.w_gate_params = quant::compute_per_channel_quant_params(lw.w_gate, inter, hidden);
  quant::quantize_weight_per_channel(lw.w_gate, inter, hidden, lw.w_gate_params, lw.w_gate_quant.data());
  
  // Up: (inter, hidden)
  lw.w_up_quant.resize(inter * hidden);
  lw.w_up_params = quant::compute_per_channel_quant_params(lw.w_up, inter, hidden);
  quant::quantize_weight_per_channel(lw.w_up, inter, hidden, lw.w_up_params, lw.w_up_quant.data());
  
  // Down: (hidden, inter)
  lw.w_down_quant.resize(hidden * inter);
  lw.w_down_params = quant::compute_per_channel_quant_params(lw.w_down, hidden, inter);
  quant::quantize_weight_per_channel(lw.w_down, hidden, inter, lw.w_down_params, lw.w_down_quant.data());
  
  lw.quantized = true;
}

// ============================================================================
// Backend-specific implementations
// ============================================================================

// ----- Baseline matmul -----
inline void do_matmul(const BackendConfig& cfg,
                      const float* a, int64_t m, int64_t k,
                      const float* w, int64_t n, int64_t wk,
                      float* out, int64_t om, int64_t on) {
  assert(k == wk && m == om && n == on);
  (void)om; (void)on;  // Suppress unused warnings
  
#ifdef __AVX2__
  if (cfg.has_simd()) {
    simd::matmul_nt(a, m, k, w, n, wk, out, m, n);
    return;
  }
#endif
  
  // Baseline
  matmul_nt(a, m, k, w, n, wk, out, m, n);
}

// ----- Quantized matmul -----
inline void do_matmul_quant(const BackendConfig& cfg,
                            const float* a, int64_t m, int64_t k,
                            const int8_t* w_quant, int64_t n,
                            const quant::PerChannelQuantParams& params,
                            float* out) {
  quant::quantized_matmul_mixed(a, m, k, w_quant, n, params, out);
}

// ----- RMSNorm -----
inline void do_rms_norm(const BackendConfig& cfg,
                        const float* x, int64_t bs, int64_t hidden,
                        const float* weight, float eps, float* out) {
#ifdef __AVX2__
  if (cfg.has_simd()) {
    simd::rms_norm_lastdim(x, bs, hidden, weight, hidden, eps, out);
    return;
  }
#endif
  
  rms_norm_lastdim(x, bs, hidden, weight, hidden, eps, out);
}

// ----- SiLU -----
inline void do_silu(const BackendConfig& cfg, float* x, int64_t n) {
#ifdef __AVX2__
  if (cfg.has_simd()) {
    // SIMD silu operates out-of-place, so we need a temp buffer or in-place
    // For now, use in-place baseline for simplicity
    simd::silu(x, n, x, n);  // simd::silu can work in-place
    return;
  }
#endif
  
  silu_inplace(x, n);
}

// ----- Softmax -----
inline void do_softmax(const BackendConfig& cfg,
                       const float* x, int64_t batch, int64_t dim,
                       float* out) {
#ifdef __AVX2__
  if (cfg.has_simd()) {
    simd::softmax_lastdim(x, batch, dim, out);
    return;
  }
#endif
  
  // Use attn_softmax_scores as a wrapper
  // Copy to output first, then softmax in place
  std::memcpy(out, x, batch * dim * sizeof(float));
  for (int64_t i = 0; i < batch; ++i) {
    float* row = out + i * dim;
    float max_val = row[0];
    for (int64_t j = 1; j < dim; ++j) max_val = std::max(max_val, row[j]);
    float sum = 0.0f;
    for (int64_t j = 0; j < dim; ++j) {
      row[j] = std::exp(row[j] - max_val);
      sum += row[j];
    }
    float inv_sum = 1.0f / sum;
    for (int64_t j = 0; j < dim; ++j) row[j] *= inv_sum;
  }
}

// ============================================================================
// Main forward function with backend dispatch
// ============================================================================

void minimind_forward_unified(
    const minimind_config& cfg,
    minimind_weights& w,
    const BackendConfig& backend,
    const int32_t* input_ids, int64_t B, int64_t S,
    const uint8_t* attention_mask, int64_t m1, int64_t m0,
    kv_cache* cache,
    minimind_workspace* ws,
    float* out_logits) {
  
  const int64_t H = cfg.n_heads;
  const int64_t KVH = cfg.n_kv_heads;
  const int64_t D = cfg.head_dim;
  const int64_t HIDDEN = cfg.hidden;
  const int64_t T = S;  // No cache for now
  const int64_t past = 0;
  
  // Quantize weights if needed
  if (backend.has_quant()) {
    for (int64_t l = 0; l < cfg.n_layers; ++l) {
      quantize_layer_weights(cfg, w.layers[l]);
    }
  }
  
  // 0) Embedding lookup
  embedding_lookup_bsh(w.tok_embedding, cfg.vocab_size, HIDDEN,
                       input_ids, B, S, ws->h0, B, S, HIDDEN);
  
  // 1) Layer stack
  for (int64_t l = 0; l < cfg.n_layers; ++l) {
    const auto& lw = w.layers[l];
    
    // === Attention ===
    
    // Pre-attn RMSNorm
    if (backend.has_fusion() && backend.has_simd()) {
      // Fused RMSNorm + Q projection
      fusion::fused_rms_norm_linear(
          ws->h0, B * S, HIDDEN,
          lw.rms_attn, cfg.rms_eps,
          lw.w_q, H * D,
          ws->attn_out_flat);
      
      // Still need K, V separately (could fuse too)
      do_rms_norm(backend, ws->h0, B * S, HIDDEN, lw.rms_attn, cfg.rms_eps, ws->h1);
      do_matmul(backend, ws->h1, B * S, HIDDEN, lw.w_k, KVH * D, HIDDEN, ws->k, B * S, KVH * D);
      do_matmul(backend, ws->h1, B * S, HIDDEN, lw.w_v, KVH * D, HIDDEN, ws->v, B * S, KVH * D);
    } else if (backend.has_quant()) {
      // Quantized path
      do_rms_norm(backend, ws->h0, B * S, HIDDEN, lw.rms_attn, cfg.rms_eps, ws->h1);
      do_matmul_quant(backend, ws->h1, B * S, HIDDEN, lw.w_q_quant.data(), H * D, lw.w_q_params, ws->attn_out_flat);
      do_matmul_quant(backend, ws->h1, B * S, HIDDEN, lw.w_k_quant.data(), KVH * D, lw.w_k_params, ws->k);
      do_matmul_quant(backend, ws->h1, B * S, HIDDEN, lw.w_v_quant.data(), KVH * D, lw.w_v_params, ws->v);
    } else {
      // Standard path (baseline or SIMD only)
      do_rms_norm(backend, ws->h0, B * S, HIDDEN, lw.rms_attn, cfg.rms_eps, ws->h1);
      do_matmul(backend, ws->h1, B * S, HIDDEN, lw.w_q, H * D, HIDDEN, ws->attn_out_flat, B * S, H * D);
      do_matmul(backend, ws->h1, B * S, HIDDEN, lw.w_k, KVH * D, HIDDEN, ws->k, B * S, KVH * D);
      do_matmul(backend, ws->h1, B * S, HIDDEN, lw.w_v, KVH * D, HIDDEN, ws->v, B * S, KVH * D);
    }
    
    // Reshape Q to (B,S,H,D)
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t s = 0; s < S; ++s) {
        const float* src = ws->attn_out_flat + (b * S + s) * (H * D);
        float* dst = ws->q + off_4d(b, s, 0, 0, S, H, D);
        std::memcpy(dst, src, H * D * sizeof(float));
      }
    }
    
    // RoPE
    apply_rotary_pos_emb_bshd(ws->q, B, S, H, D, ws->k, B, S, KVH, D,
                              w.rope_cos + past * D, w.rope_sin + past * D, S, D);
    
    // Repeat KV
    repeat_kv_btkd_to_bthd(ws->k, B, T, KVH, D, H / KVH, ws->k_rep, B, T, H, D);
    repeat_kv_btkd_to_bthd(ws->v, B, T, KVH, D, H / KVH, ws->v_rep, B, T, H, D);
    
    // Transpose
    transpose_bshd_to_bhsd(ws->q, B, S, H, D, ws->q_bhsd, B, H, S, D);
    transpose_bthd_to_bhtd(ws->k_rep, B, T, H, D, ws->k_bhtd, B, H, T, D);
    transpose_bthd_to_bhtd(ws->v_rep, B, T, H, D, ws->v_bhtd, B, H, T, D);
    
    // Attention scores
    attn_qk_matmul(ws->q_bhsd, B, H, S, D, ws->k_bhtd, B, H, T, D,
                   1.0f / std::sqrt(static_cast<float>(D)),
                   ws->scores, B, H, S, T);
    
    // Causal mask
    apply_causal_mask(ws->scores, B, H, S, T, past);
    
    // Attention mask
    if (attention_mask != nullptr) {
      add_attention_mask(ws->scores, B, H, S, T, attention_mask, m1, m0);
    }
    
    // Softmax
    attn_softmax_scores(ws->scores, B, H, S, T, ws->probs, B, H, S, T);
    
    // Attn @ V
    attn_pv_matmul(ws->probs, B, H, S, T, ws->v_bhtd, B, H, T, D,
                   ws->attn_out, B, H, S, D);
    
    // Transpose back
    transpose_bhsd_to_bshd(ws->attn_out, B, H, S, D, ws->attn_out_bshd, B, S, H, D);
    std::memcpy(ws->attn_out_flat, ws->attn_out_bshd, B * S * H * D * sizeof(float));
    
    // O projection
    if (backend.has_quant()) {
      do_matmul_quant(backend, ws->attn_out_flat, B * S, H * D,
                      lw.w_o_quant.data(), HIDDEN, lw.w_o_params, ws->h1);
    } else {
      do_matmul(backend, ws->attn_out_flat, B * S, H * D, lw.w_o, HIDDEN, H * D, ws->h1, B * S, HIDDEN);
    }
    
    // Residual
    for (int64_t i = 0; i < B * S * HIDDEN; ++i) {
      ws->h1[i] = ws->h0[i] + ws->h1[i];
    }
    
    // === FFN ===
    
    if (backend.has_fusion() && backend.has_simd()) {
      // Fused FFN: RMSNorm -> Gate+Up -> SwiGLU -> Down -> Residual
      fusion::fused_ffn_block(
          ws->h1, B * S, HIDDEN,
          lw.rms_ffn, cfg.rms_eps,
          lw.w_gate, lw.w_up, lw.w_down, cfg.inter,
          ws->h0, ws->ffn_mid);
    } else if (backend.has_quant()) {
      // Quantized FFN
      do_rms_norm(backend, ws->h1, B * S, HIDDEN, lw.rms_ffn, cfg.rms_eps, ws->ffn_out);
      do_matmul_quant(backend, ws->ffn_out, B * S, HIDDEN, lw.w_gate_quant.data(), cfg.inter, lw.w_gate_params, ws->ffn_gate);
      do_matmul_quant(backend, ws->ffn_out, B * S, HIDDEN, lw.w_up_quant.data(), cfg.inter, lw.w_up_params, ws->ffn_up);
      do_silu(backend, ws->ffn_gate, B * S * cfg.inter);
      mul(ws->ffn_gate, B * S * cfg.inter, ws->ffn_up, B * S * cfg.inter, ws->ffn_mid, B * S * cfg.inter);
      do_matmul_quant(backend, ws->ffn_mid, B * S, cfg.inter, lw.w_down_quant.data(), HIDDEN, lw.w_down_params, ws->ffn_out);
      std::memcpy(ws->h0, ws->h1, B * S * HIDDEN * sizeof(float));
      for (int64_t i = 0; i < B * S * HIDDEN; ++i) ws->h0[i] += ws->ffn_out[i];
    } else {
      // Standard FFN
      std::memcpy(ws->h0, ws->h1, B * S * HIDDEN * sizeof(float));
      do_rms_norm(backend, ws->h1, B * S, HIDDEN, lw.rms_ffn, cfg.rms_eps, ws->ffn_out);
      do_matmul(backend, ws->ffn_out, B * S, HIDDEN, lw.w_gate, cfg.inter, HIDDEN, ws->ffn_gate, B * S, cfg.inter);
      do_matmul(backend, ws->ffn_out, B * S, HIDDEN, lw.w_up, cfg.inter, HIDDEN, ws->ffn_up, B * S, cfg.inter);
      do_silu(backend, ws->ffn_gate, B * S * cfg.inter);
      mul(ws->ffn_gate, B * S * cfg.inter, ws->ffn_up, B * S * cfg.inter, ws->ffn_mid, B * S * cfg.inter);
      do_matmul(backend, ws->ffn_mid, B * S, cfg.inter, lw.w_down, HIDDEN, cfg.inter, ws->ffn_out, B * S, HIDDEN);
      for (int64_t i = 0; i < B * S * HIDDEN; ++i) ws->h0[i] += ws->ffn_out[i];
    }
  }
  
  // 2) Final norm
  do_rms_norm(backend, ws->h0, B * S, HIDDEN, w.final_rms, cfg.rms_eps, ws->h1);
  
  // 3) LM head
  do_matmul(backend, ws->h1, B * S, HIDDEN, w.lm_head, cfg.vocab_size, HIDDEN,
            out_logits, B * S, cfg.vocab_size);
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
  std::string backend_str = (argc >= 4) ? argv[3] : "baseline";
  
  BackendConfig backend = BackendConfig::from_string(backend_str);
  
  std::cout << "=== MiniMind Unified Inference ===\n";
  std::cout << "Backend: " << backend.feature_string() << "\n\n";
  
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
  const int64_t vocab = cfg_j["vocab_size"];
  const int64_t hidden = cfg_j["hidden_size"];
  const int64_t layers = cfg_j["num_hidden_layers"];
  const int64_t heads = cfg_j["num_attention_heads"];
  const int64_t kv_heads = cfg_j["num_key_value_heads"];
  const int64_t inter = cfg_j["intermediate_size"];
  const int64_t max_pos = cfg_j["max_position_embeddings"];
  const int64_t head_dim = hidden / heads;
  
  auto meta = j["meta"];
  const int64_t B = meta["B"];
  const int64_t S = meta["S"];
  
  std::cout << "Config: hidden=" << hidden << " layers=" << layers
            << " heads=" << heads << " vocab=" << vocab << "\n";
  std::cout << "Input: B=" << B << " S=" << S << "\n\n";
  
  // Load inputs
  std::string input_ids_b64 = j["inputs"]["input_ids"]["data"];
  std::vector<uint8_t> input_ids_bytes = b64_decode(input_ids_b64);
  std::vector<int32_t> input_ids(input_ids_bytes.size() / sizeof(int32_t));
  std::memcpy(input_ids.data(), input_ids_bytes.data(), input_ids_bytes.size());
  
  std::string attn_mask_b64 = j["inputs"]["attention_mask"]["data"];
  std::vector<uint8_t> attn_mask = b64_decode(attn_mask_b64);
  
  // Load weights
  std::vector<float> rope_cos = load_tensor_from_json(j["rope"]["cos"]);
  std::vector<float> rope_sin = load_tensor_from_json(j["rope"]["sin"]);
  std::vector<float> tok_embedding = load_tensor_from_json(j["weights"]["tok_embedding"]);
  std::vector<float> final_rms = load_tensor_from_json(j["weights"]["final_rms"]);
  std::vector<float> lm_head = load_tensor_from_json(j["weights"]["lm_head"]);
  
  // Load layer weights
  std::vector<minimind_layer_weights> layer_w(layers);
  std::vector<std::vector<float>> wq(layers), wk(layers), wv(layers), wo(layers);
  std::vector<std::vector<float>> wgate(layers), wup(layers), wdown(layers);
  std::vector<std::vector<float>> rms_attn(layers), rms_ffn(layers);
  
  auto layers_j = j["weights"]["layers"];
  for (int64_t l = 0; l < layers; ++l) {
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
    
    layer_w[l].rms_attn = rms_attn[l].data();
    layer_w[l].rms_ffn = rms_ffn[l].data();
    layer_w[l].w_q = wq[l].data();
    layer_w[l].w_k = wk[l].data();
    layer_w[l].w_v = wv[l].data();
    layer_w[l].w_o = wo[l].data();
    layer_w[l].w_gate = wgate[l].data();
    layer_w[l].w_up = wup[l].data();
    layer_w[l].w_down = wdown[l].data();
  }
  
  // Build config
  minimind_config cfg;
  cfg.vocab_size = vocab;
  cfg.hidden = hidden;
  cfg.n_layers = layers;
  cfg.n_heads = heads;
  cfg.n_kv_heads = kv_heads;
  cfg.head_dim = head_dim;
  cfg.inter = inter;
  cfg.rms_eps = 1e-5f;
  cfg.max_pos = max_pos;
  
  minimind_weights W;
  W.tok_embedding = tok_embedding.data();
  W.final_rms = final_rms.data();
  W.lm_head = lm_head.data();
  W.rope_cos = rope_cos.data();
  W.rope_sin = rope_sin.data();
  W.layers = layer_w.data();
  
  // Allocate workspace
  const int64_t T = S;
  auto h0 = std::make_unique<float[]>(B * S * hidden);
  auto h1 = std::make_unique<float[]>(B * S * hidden);
  auto q = std::make_unique<float[]>(B * S * heads * head_dim);
  auto k = std::make_unique<float[]>(B * S * kv_heads * head_dim);
  auto v = std::make_unique<float[]>(B * S * kv_heads * head_dim);
  auto k_rep = std::make_unique<float[]>(B * T * heads * head_dim);
  auto v_rep = std::make_unique<float[]>(B * T * heads * head_dim);
  auto q_bhsd = std::make_unique<float[]>(B * heads * S * head_dim);
  auto k_bhtd = std::make_unique<float[]>(B * heads * T * head_dim);
  auto v_bhtd = std::make_unique<float[]>(B * heads * T * head_dim);
  auto scores = std::make_unique<float[]>(B * heads * S * T);
  auto probs = std::make_unique<float[]>(B * heads * S * T);
  auto attn_out = std::make_unique<float[]>(B * heads * S * head_dim);
  auto attn_out_bshd = std::make_unique<float[]>(B * S * heads * head_dim);
  auto attn_out_flat = std::make_unique<float[]>(B * S * heads * head_dim);
  auto ffn_gate = std::make_unique<float[]>(B * S * inter);
  auto ffn_up = std::make_unique<float[]>(B * S * inter);
  auto ffn_mid = std::make_unique<float[]>(B * S * inter);
  auto ffn_out = std::make_unique<float[]>(B * S * hidden);
  auto logits = std::make_unique<float[]>(B * S * vocab);
  
  minimind_workspace ws;
  ws.h0 = h0.get();
  ws.h1 = h1.get();
  ws.q = q.get();
  ws.k = k.get();
  ws.v = v.get();
  ws.k_rep = k_rep.get();
  ws.v_rep = v_rep.get();
  ws.q_bhsd = q_bhsd.get();
  ws.k_bhtd = k_bhtd.get();
  ws.v_bhtd = v_bhtd.get();
  ws.scores = scores.get();
  ws.probs = probs.get();
  ws.attn_out = attn_out.get();
  ws.attn_out_bshd = attn_out_bshd.get();
  ws.attn_out_flat = attn_out_flat.get();
  ws.ffn_gate = ffn_gate.get();
  ws.ffn_up = ffn_up.get();
  ws.ffn_mid = ffn_mid.get();
  ws.ffn_out = ffn_out.get();
  ws.logits = logits.get();
  
  // Warmup
  minimind_forward_unified(cfg, W, backend, input_ids.data(), B, S,
                          attn_mask.data(), B, T, nullptr, &ws, logits.get());
  
  // Timed run
  const int num_runs = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; ++i) {
    minimind_forward_unified(cfg, W, backend, input_ids.data(), B, S,
                            attn_mask.data(), B, T, nullptr, &ws, logits.get());
  }
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
  
  // Stats
  float min_logit = logits[0], max_logit = logits[0];
  double mean_logit = 0.0;
  for (int64_t i = 0; i < B * S * vocab; ++i) {
    min_logit = std::min(min_logit, logits[i]);
    max_logit = std::max(max_logit, logits[i]);
    mean_logit += logits[i];
  }
  mean_logit /= (B * S * vocab);
  
  std::cout << "========================================\n";
  std::cout << "[Forward] Shape: (" << B << ", " << S << ", " << vocab << ")\n";
  std::cout << "[Backend] " << backend.feature_string() << "\n";
  std::cout << "[Timing] " << elapsed_ms << " ms (avg of " << num_runs << " runs)\n";
  std::cout << "[Logits] Min: " << min_logit << ", Max: " << max_logit
            << ", Mean: " << mean_logit << "\n";
  std::cout << "========================================\n";
  
  // Save logits
  std::string suffix = backend.feature_string();
  std::replace(suffix.begin(), suffix.end(), '+', '_');
  std::string logits_path = dump_dir + "/logits_" + suffix + ".npy";
  write_f32(logits_path, logits.get(), B * S * vocab);
  std::cout << "[OK] Saved: " << logits_path << "\n";
  
  return 0;
}

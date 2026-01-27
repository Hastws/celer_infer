// minimind_extreme.cpp - Extreme Performance CPU Inference
// ============================================================================
// Maximum performance CPU implementation combining ALL optimizations:
//   - OpenMP multi-threading
//   - AVX2/FMA SIMD vectorization
//   - Cache blocking/tiling for GEMM
//   - Memory prefetching
//   - Fused operations
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <omp.h>

// Include extreme ops (OpenMP + SIMD + Tiling)
#include "tensor_op_extreme.hpp"

// For JSON loading
#include "tensor_op.hpp"  // base utilities
#include "nlohmann/json.hpp"

using namespace celer_infer;
using json = nlohmann::json;

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
};

struct minimind_weights {
  const float* tok_embedding = nullptr;
  const float* final_rms = nullptr;
  const float* lm_head = nullptr;
  const float* rope_cos = nullptr;
  const float* rope_sin = nullptr;
  std::vector<minimind_layer_weights> layers;
};

// ============================================================================
// Workspace with aligned allocations
// ============================================================================

struct minimind_workspace {
  float* h0 = nullptr;
  float* h1 = nullptr;
  float* q = nullptr;
  float* k = nullptr;
  float* v = nullptr;
  float* k_rep = nullptr;
  float* v_rep = nullptr;
  float* q_bhsd = nullptr;
  float* k_bhsd = nullptr;
  float* v_bhsd = nullptr;
  float* scores = nullptr;
  float* attn_out = nullptr;
  float* attn_out_bshd = nullptr;
  float* ffn_gate = nullptr;
  float* ffn_up = nullptr;
  float* ffn_out = nullptr;
  float* logits = nullptr;
  
  void allocate(const minimind_config& cfg, int64_t B, int64_t S) {
    int64_t H = cfg.n_heads;
    int64_t KVH = cfg.n_kv_heads;
    int64_t D = cfg.head_dim;
    int64_t T = S;
    
    // Use aligned allocation for SIMD
    auto aligned_alloc_f = [](size_t n) -> float* {
      void* ptr = nullptr;
      posix_memalign(&ptr, 64, n * sizeof(float));  // 64-byte aligned
      return static_cast<float*>(ptr);
    };
    
    h0 = aligned_alloc_f(B * S * cfg.hidden);
    h1 = aligned_alloc_f(B * S * cfg.hidden);
    q = aligned_alloc_f(B * S * H * D);
    k = aligned_alloc_f(B * S * KVH * D);
    v = aligned_alloc_f(B * S * KVH * D);
    k_rep = aligned_alloc_f(B * T * H * D);
    v_rep = aligned_alloc_f(B * T * H * D);
    q_bhsd = aligned_alloc_f(B * H * S * D);
    k_bhsd = aligned_alloc_f(B * H * T * D);
    v_bhsd = aligned_alloc_f(B * H * T * D);
    scores = aligned_alloc_f(B * H * S * T);
    attn_out = aligned_alloc_f(B * H * S * D);
    attn_out_bshd = aligned_alloc_f(B * S * H * D);
    ffn_gate = aligned_alloc_f(B * S * cfg.inter);
    ffn_up = aligned_alloc_f(B * S * cfg.inter);
    ffn_out = aligned_alloc_f(B * S * cfg.hidden);
    logits = aligned_alloc_f(B * S * cfg.vocab_size);
  }
  
  void free_all() {
    free(h0); free(h1); free(q); free(k); free(v);
    free(k_rep); free(v_rep);
    free(q_bhsd); free(k_bhsd); free(v_bhsd);
    free(scores); free(attn_out); free(attn_out_bshd);
    free(ffn_gate); free(ffn_up); free(ffn_out);
    free(logits);
  }
};

// ============================================================================
// Extreme Performance Forward Pass
// ============================================================================

void minimind_forward_extreme(
    const minimind_config& cfg,
    const minimind_weights& w,
    const int32_t* input_ids,
    int64_t B, int64_t S,
    minimind_workspace& ws,
    float* logits_out) {
  
  const int64_t H = cfg.n_heads;
  const int64_t KVH = cfg.n_kv_heads;
  const int64_t D = cfg.head_dim;
  const int64_t T = S;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int64_t n_rep = H / KVH;
  (void)n_rep;  // Suppress unused warning
  
  // 1. Embedding lookup (parallel)
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < B * S; ++i) {
    int32_t idx = input_ids[i];
    std::memcpy(ws.h0 + i * cfg.hidden, 
                w.tok_embedding + idx * cfg.hidden,
                cfg.hidden * sizeof(float));
  }
  
  // 2. Layer stack
  for (int64_t l = 0; l < cfg.n_layers; ++l) {
    const auto& lw = w.layers[l];
    
    // Pre-attention RMS Norm (parallel)
    extreme::rms_norm_parallel(ws.h0, B * S, cfg.hidden, lw.rms_attn, cfg.rms_eps, ws.h1);
    
    // Q, K, V projections (parallel tiled GEMM)
    extreme::gemm_extreme(ws.h1, B * S, cfg.hidden, lw.w_q, H * D, ws.q);
    extreme::gemm_extreme(ws.h1, B * S, cfg.hidden, lw.w_k, KVH * D, ws.k);
    extreme::gemm_extreme(ws.h1, B * S, cfg.hidden, lw.w_v, KVH * D, ws.v);
    
    // RoPE (parallel) - using rotate_half style
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < B * S * H; ++idx) {
      int64_t h = idx % H;
      int64_t s = (idx / H) % S;
      int64_t b = idx / (H * S);
      
      float* q_ptr = ws.q + ((b * S + s) * H + h) * D;
      alignas(32) float q_copy[256];
      std::memcpy(q_copy, q_ptr, D * sizeof(float));
      
      int64_t half_D = D / 2;
      for (int64_t d = 0; d < D; ++d) {
        float cos_val = w.rope_cos[s * D + d];
        float sin_val = w.rope_sin[s * D + d];
        float q_d = q_copy[d];
        float q_rot = (d < half_D) ? -q_copy[d + half_D] : q_copy[d - half_D];
        q_ptr[d] = q_d * cos_val + q_rot * sin_val;
      }
    }
    
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < B * S * KVH; ++idx) {
      int64_t h = idx % KVH;
      int64_t s = (idx / KVH) % S;
      int64_t b = idx / (KVH * S);
      
      float* k_ptr = ws.k + ((b * S + s) * KVH + h) * D;
      alignas(32) float k_copy[256];
      std::memcpy(k_copy, k_ptr, D * sizeof(float));
      
      int64_t half_D = D / 2;
      for (int64_t d = 0; d < D; ++d) {
        float cos_val = w.rope_cos[s * D + d];
        float sin_val = w.rope_sin[s * D + d];
        float k_d = k_copy[d];
        float k_rot = (d < half_D) ? -k_copy[d + half_D] : k_copy[d - half_D];
        k_ptr[d] = k_d * cos_val + k_rot * sin_val;
      }
    }
    
    // Repeat KV (parallel)
    extreme::repeat_kv_parallel(ws.k, ws.k_rep, B, S, KVH, H, D);
    extreme::repeat_kv_parallel(ws.v, ws.v_rep, B, S, KVH, H, D);
    
    // Transpose to BHSD (parallel)
    extreme::transpose_bshd_to_bhsd_parallel(ws.q, ws.q_bhsd, B, S, H, D);
    extreme::transpose_bshd_to_bhsd_parallel(ws.k_rep, ws.k_bhsd, B, T, H, D);
    extreme::transpose_bshd_to_bhsd_parallel(ws.v_rep, ws.v_bhsd, B, T, H, D);
    
    // Batched Q @ K^T (parallel over B*H)
    extreme::batched_gemm_parallel(ws.q_bhsd, ws.k_bhsd, ws.scores, B * H, S, D, T);
    
    // Causal softmax (parallel)
    extreme::causal_softmax_parallel(ws.scores, B, H, S, T, scale);
    
    // Attention @ V (parallel over B*H)
    #pragma omp parallel for schedule(dynamic)
    for (int64_t bh = 0; bh < B * H; ++bh) {
      const float* scores_bh = ws.scores + bh * S * T;
      const float* v_bh = ws.v_bhsd + bh * T * D;
      float* out_bh = ws.attn_out + bh * S * D;
      
      // scores: (S, T) @ v: (T, D) -> out: (S, D)
      for (int64_t s = 0; s < S; ++s) {
        for (int64_t d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (int64_t t = 0; t < T; ++t) {
            sum += scores_bh[s * T + t] * v_bh[t * D + d];
          }
          out_bh[s * D + d] = sum;
        }
      }
    }
    
    // Transpose back to BSHD (parallel)
    extreme::transpose_bhsd_to_bshd_parallel(ws.attn_out, ws.attn_out_bshd, B, H, S, D);
    
    // O projection (parallel GEMM)
    extreme::gemm_extreme(ws.attn_out_bshd, B * S, H * D, lw.w_o, cfg.hidden, ws.h1);
    
    // Residual add (parallel)
    extreme::add_parallel(ws.h0, ws.h1, B * S * cfg.hidden, ws.h0);
    
    // FFN: Pre-FFN RMS Norm
    extreme::rms_norm_parallel(ws.h0, B * S, cfg.hidden, lw.rms_ffn, cfg.rms_eps, ws.h1);
    
    // Gate and Up projections (parallel GEMM)
    extreme::gemm_extreme(ws.h1, B * S, cfg.hidden, lw.w_gate, cfg.inter, ws.ffn_gate);
    extreme::gemm_extreme(ws.h1, B * S, cfg.hidden, lw.w_up, cfg.inter, ws.ffn_up);
    
    // SwiGLU: SiLU(gate) * up (parallel)
    extreme::swiglu_parallel(ws.ffn_gate, ws.ffn_up, B * S * cfg.inter, ws.ffn_gate);
    
    // Down projection (parallel GEMM)
    extreme::gemm_extreme(ws.ffn_gate, B * S, cfg.inter, lw.w_down, cfg.hidden, ws.ffn_out);
    
    // Residual add (parallel)
    extreme::add_parallel(ws.h0, ws.ffn_out, B * S * cfg.hidden, ws.h0);
  }
  
  // 3. Final RMS Norm
  extreme::rms_norm_parallel(ws.h0, B * S, cfg.hidden, w.final_rms, cfg.rms_eps, ws.h1);
  
  // 4. LM Head (parallel GEMM)
  extreme::gemm_extreme(ws.h1, B * S, cfg.hidden, w.lm_head, cfg.vocab_size, ws.logits);
  
  // Copy output
  std::memcpy(logits_out, ws.logits, B * S * cfg.vocab_size * sizeof(float));
}

// ============================================================================
// JSON Loading - Base64 Decoder
// ============================================================================

static int b64_char_to_val(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

static std::vector<uint8_t> b64_decode(const std::string& b64) {
  std::vector<uint8_t> ret;
  int val = 0;
  int bits = 0;
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

static std::vector<float> load_tensor(const json& j) {
  std::string b64 = j["data"];
  std::vector<uint8_t> decoded = b64_decode(b64);
  std::vector<float> result(decoded.size() / sizeof(float));
  std::memcpy(result.data(), decoded.data(), decoded.size());
  return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  std::string json_path = (argc >= 2) ? argv[1] : "dump_minimind/minimind.json";
  std::string dump_dir = (argc >= 3) ? argv[2] : "dump_minimind";
  
  // Configure OpenMP
  int num_threads = omp_get_max_threads();
  std::cout << "=== MiniMind Extreme CPU Inference ===" << std::endl;
  std::cout << "OpenMP threads: " << num_threads << std::endl << std::endl;
  
  // Load JSON
  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    std::cerr << "Failed to open: " << json_path << std::endl;
    return 1;
  }
  json j;
  json_file >> j;
  json_file.close();
  
  // Parse config
  auto cfg_j = j["config"];
  minimind_config cfg;
  cfg.vocab_size = cfg_j["vocab_size"].get<int64_t>();
  cfg.hidden = cfg_j["hidden_size"].get<int64_t>();
  cfg.n_layers = cfg_j["num_hidden_layers"].get<int64_t>();
  cfg.n_heads = cfg_j["num_attention_heads"].get<int64_t>();
  cfg.n_kv_heads = cfg_j["num_key_value_heads"].get<int64_t>();
  // head_dim is computed from hidden_size / n_heads
  cfg.head_dim = cfg.hidden / cfg.n_heads;
  cfg.inter = cfg_j["intermediate_size"].get<int64_t>();
  cfg.rms_eps = cfg_j.value("rms_norm_eps", 1e-5f);
  cfg.max_pos = cfg_j["max_position_embeddings"].get<int64_t>();
  
  std::cout << "Config: vocab=" << cfg.vocab_size 
            << ", hidden=" << cfg.hidden
            << ", layers=" << cfg.n_layers
            << ", heads=" << cfg.n_heads << "/" << cfg.n_kv_heads 
            << ", head_dim=" << cfg.head_dim << std::endl << std::endl;
  
  // Load weights
  auto& weights_j = j["weights"];
  
  std::vector<float> tok_emb = load_tensor(weights_j["tok_embedding"]);
  std::vector<float> final_rms = load_tensor(weights_j["final_rms"]);
  std::vector<float> lm_head = load_tensor(weights_j["lm_head"]);
  std::vector<float> rope_cos = load_tensor(j["rope"]["cos"]);
  std::vector<float> rope_sin = load_tensor(j["rope"]["sin"]);
  
  minimind_weights w;
  w.tok_embedding = tok_emb.data();
  w.final_rms = final_rms.data();
  w.lm_head = lm_head.data();
  w.rope_cos = rope_cos.data();
  w.rope_sin = rope_sin.data();
  
  // Load layer weights - using array format
  auto& layers_j = weights_j["layers"];
  std::vector<std::vector<float>> layer_data(cfg.n_layers * 9);
  w.layers.resize(cfg.n_layers);
  
  for (int64_t l = 0; l < cfg.n_layers; ++l) {
    auto& layer_j = layers_j[l];
    int base = l * 9;
    
    layer_data[base + 0] = load_tensor(layer_j["rms_attn"]);
    layer_data[base + 1] = load_tensor(layer_j["rms_ffn"]);
    layer_data[base + 2] = load_tensor(layer_j["wq"]);
    layer_data[base + 3] = load_tensor(layer_j["wk"]);
    layer_data[base + 4] = load_tensor(layer_j["wv"]);
    layer_data[base + 5] = load_tensor(layer_j["wo"]);
    layer_data[base + 6] = load_tensor(layer_j["w_gate"]);
    layer_data[base + 7] = load_tensor(layer_j["w_up"]);
    layer_data[base + 8] = load_tensor(layer_j["w_down"]);
    
    w.layers[l].rms_attn = layer_data[base + 0].data();
    w.layers[l].rms_ffn = layer_data[base + 1].data();
    w.layers[l].w_q = layer_data[base + 2].data();
    w.layers[l].w_k = layer_data[base + 3].data();
    w.layers[l].w_v = layer_data[base + 4].data();
    w.layers[l].w_o = layer_data[base + 5].data();
    w.layers[l].w_gate = layer_data[base + 6].data();
    w.layers[l].w_up = layer_data[base + 7].data();
    w.layers[l].w_down = layer_data[base + 8].data();
  }
  
  // Load input - from inputs/input_ids
  std::string input_ids_b64 = j["inputs"]["input_ids"]["data"];
  std::vector<uint8_t> input_ids_bytes = b64_decode(input_ids_b64);
  std::vector<int32_t> input_ids_vec(input_ids_bytes.size() / sizeof(int32_t));
  std::memcpy(input_ids_vec.data(), input_ids_bytes.data(), input_ids_bytes.size());
  
  int64_t B = 1;
  int64_t S = static_cast<int64_t>(input_ids_vec.size());
  
  std::cout << "Input: B=" << B << ", S=" << S << std::endl << std::endl;
  
  // Allocate workspace
  minimind_workspace ws;
  ws.allocate(cfg, B, S);
  
  std::vector<float> logits(B * S * cfg.vocab_size);
  
  // Warmup
  minimind_forward_extreme(cfg, w, input_ids_vec.data(), B, S, ws, logits.data());
  
  // Benchmark
  int n_runs = 100;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n_runs; ++i) {
    minimind_forward_extreme(cfg, w, input_ids_vec.data(), B, S, ws, logits.data());
  }
  auto end = std::chrono::high_resolution_clock::now();
  
  double ms = std::chrono::duration<double, std::milli>(end - start).count() / n_runs;
  std::cout << "Extreme Inference: " << ms << " ms/forward (" << num_threads << " threads)" << std::endl;
  
  // Write output
  std::string out_path = dump_dir + "/logits_extreme.bin";
  std::ofstream ofs(out_path, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
  std::cout << "Output written to: " << out_path << std::endl;
  
  ws.free_all();
  return 0;
}

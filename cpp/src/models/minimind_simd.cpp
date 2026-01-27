// minimind_simd.cpp - SIMD-optimized MiniMind inference
// ============================================================================
// Uses AVX2-optimized tensor operations for improved performance.
// ============================================================================

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdlib>

// First include the baseline tensor_op.hpp for common functions
#include "tensor_op.hpp"

// Then include SIMD-optimized versions (will use simd:: namespace)
#include "tensor_op_simd.hpp"

#include "nlohmann/json.hpp"

using namespace celer_infer;

std::string g_dump_dir;

// ============================
// Model wiring structs (same as baseline)
// ============================

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
  int64_t moe_topk = 2;
  int64_t moe_experts = 0;
  int64_t moe_shared = 0;
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
  const float* b_q = nullptr;
  const float* b_k = nullptr;
  const float* b_v = nullptr;
  const float* b_o = nullptr;
  const float* b_gate = nullptr;
  const float* b_up = nullptr;
  const float* b_down = nullptr;
  const float* moe_gate_w = nullptr;
};

struct minimind_weights {
  const float* tok_embedding = nullptr;
  const float* final_rms = nullptr;
  const float* lm_head = nullptr;
  const float* rope_cos = nullptr;
  const float* rope_sin = nullptr;
  const minimind_layer_weights* layers = nullptr;
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

// Helper offset function
inline int64_t off_4d_bshd(int64_t b, int64_t s, int64_t h, int64_t d,
                           int64_t S, int64_t H, int64_t D) {
  return celer_infer::off_4d(b, s, h, d, S, H, D);
}

// ============================
// SIMD-optimized forward functions
// ============================

inline void minimind_attention_forward_simd(
    const minimind_config& cfg, const minimind_layer_weights& lw,
    const float* rope_cos, const float* rope_sin, int64_t max_pos,
    const uint8_t* attn_mask_bt, int64_t m1, int64_t m0,
    kv_cache* cache, int layer_id, const float* x, int64_t B, int64_t S,
    int64_t HIDDEN, minimind_workspace* ws, float* out) {
  
  assert(HIDDEN == cfg.hidden);
  assert(cfg.head_dim * cfg.n_heads == cfg.hidden);
  const int64_t H = cfg.n_heads;
  const int64_t KVH = cfg.n_kv_heads;
  const int64_t D = cfg.head_dim;

  // 1) pre-attn rmsnorm (SIMD optimized)
  simd::rms_norm_lastdim(x, B * S, HIDDEN, lw.rms_attn, HIDDEN, cfg.rms_eps, ws->h1);

  // 2) q,k,v projections (SIMD optimized matmul_nt)
  simd::matmul_nt(ws->h1, B * S, HIDDEN, lw.w_q, H * D, HIDDEN, ws->attn_out_flat,
                  B * S, H * D);
  if (lw.b_q)
    add_bias_lastdim_inplace(ws->attn_out_flat, B * S, H * D, lw.b_q, H * D);

  simd::matmul_nt(ws->h1, B * S, HIDDEN, lw.w_k, KVH * D, HIDDEN, ws->k, B * S,
                  KVH * D);
  if (lw.b_k)
    add_bias_lastdim_inplace(ws->k, B * S, KVH * D, lw.b_k, KVH * D);

  simd::matmul_nt(ws->h1, B * S, HIDDEN, lw.w_v, KVH * D, HIDDEN, ws->v, B * S,
                  KVH * D);
  if (lw.b_v)
    add_bias_lastdim_inplace(ws->v, B * S, KVH * D, lw.b_v, KVH * D);

  // 3) view to (B,S,H,D) and (B,S,KVH,D)
  {
    const float* q_src = ws->attn_out_flat;
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t s = 0; s < S; ++s) {
        const float* src = q_src + (b * S + s) * (H * D);
        float* dst = ws->q + off_4d_bshd(b, s, 0, 0, S, H, D);
        std::memcpy(dst, src, static_cast<size_t>(H * D) * sizeof(float));
      }
    }
  }

  // 4) RoPE (use baseline - not SIMD critical)
  const int64_t past = (cache && cache->cur_t > 0) ? cache->cur_t : 0;
  assert(past + S <= max_pos);
  apply_rotary_pos_emb_bshd(ws->q, B, S, H, D, ws->k, B, S, KVH, D,
                            rope_cos + past * D, rope_sin + past * D, S, D);

  // 5) KV cache
  const int64_t T = past + S;
  if (cache != nullptr) {
    assert(cache->max_t >= T);
    kv_cache_append_btkd(cache->k[layer_id], B, cache->max_t, KVH, D, ws->k, B,
                         S, KVH, D, past);
    kv_cache_append_btkd(cache->v[layer_id], B, cache->max_t, KVH, D, ws->v, B,
                         S, KVH, D, past);
  }

  // 6) Build k_cat/v_cat views
  const float* k_cat = (cache != nullptr) ? cache->k[layer_id] : ws->k;
  const float* v_cat = (cache != nullptr) ? cache->v[layer_id] : ws->v;
  const int64_t cat_T = (cache != nullptr) ? T : S;

  // 7) repeat kv (baseline, not SIMD critical) and transpose
  repeat_kv_btkd_to_bthd(k_cat, B, cat_T, KVH, D, cfg.n_heads / cfg.n_kv_heads,
                         ws->k_rep, B, cat_T, H, D);
  repeat_kv_btkd_to_bthd(v_cat, B, cat_T, KVH, D, cfg.n_heads / cfg.n_kv_heads,
                         ws->v_rep, B, cat_T, H, D);

  transpose_bshd_to_bhsd(ws->q, B, S, H, D, ws->q_bhsd, B, H, S, D);
  transpose_bthd_to_bhtd(ws->k_rep, B, cat_T, H, D, ws->k_bhtd, B, H, cat_T, D);
  transpose_bthd_to_bhtd(ws->v_rep, B, cat_T, H, D, ws->v_bhtd, B, H, cat_T, D);

  // 8) attention: scores = q @ k^T (SIMD optimized)
  simd::attn_qk_matmul(ws->q_bhsd, B, H, S, D, ws->k_bhtd, B, H, cat_T, D,
                       ws->scores, B, H, S, cat_T);
  // Scale
  simd::scale(ws->scores, B * H * S * cat_T, 1.0f / std::sqrt(static_cast<float>(D)));

  // 9) causal mask
  simd::apply_causal_mask(ws->scores, B, H, S, cat_T);

  // 10) padding mask
  if (attn_mask_bt != nullptr) {
    assert(m1 == B && m0 == cat_T);
    add_attention_mask(ws->scores, B, H, S, cat_T, attn_mask_bt, m1, m0);
  }

  // 11) softmax (SIMD optimized)
  simd::softmax_lastdim(ws->scores, B * H * S, cat_T, ws->probs);

  // 12) attn_out = probs @ v (SIMD optimized)
  simd::attn_pv_matmul(ws->probs, B, H, S, cat_T, ws->v_bhtd, B, H, cat_T, D,
                       ws->attn_out, B, H, S, D);

  // 13) transpose and merge heads
  transpose_bhsd_to_bshd(ws->attn_out, B, H, S, D, ws->attn_out_bshd, B, S, H, D);
  std::memcpy(ws->attn_out_flat, ws->attn_out_bshd,
              static_cast<size_t>(B * S * H * D) * sizeof(float));

  // 14) o_proj (SIMD optimized)
  simd::matmul_nt(ws->attn_out_flat, B * S, H * D, lw.w_o, HIDDEN, H * D,
                  ws->h1, B * S, HIDDEN);
  if (lw.b_o)
    add_bias_lastdim_inplace(ws->h1, B * S, HIDDEN, lw.b_o, HIDDEN);

  // 15) residual (SIMD optimized)
  simd::copy(x, B * S * HIDDEN, out, B * S * HIDDEN);
  simd::add(out, B * S * HIDDEN, ws->h1, B * S * HIDDEN, out, B * S * HIDDEN);
}

inline void minimind_ffn_forward_simd(const minimind_config& cfg,
                                      const minimind_layer_weights& lw,
                                      const float* x, int64_t B, int64_t S,
                                      int64_t HIDDEN, minimind_workspace* ws,
                                      float* out) {
  // Save original input
  simd::copy(x, B * S * HIDDEN, out, B * S * HIDDEN);

  // pre-ffn rmsnorm (SIMD optimized)
  simd::rms_norm_lastdim(x, B * S, HIDDEN, lw.rms_ffn, HIDDEN, cfg.rms_eps, ws->h1);

  // gate/up projections (SIMD optimized)
  simd::matmul_nt(ws->h1, B * S, HIDDEN, lw.w_gate, cfg.inter, HIDDEN, ws->ffn_gate,
                  B * S, cfg.inter);
  if (lw.b_gate)
    add_bias_lastdim_inplace(ws->ffn_gate, B * S, cfg.inter, lw.b_gate, cfg.inter);

  simd::matmul_nt(ws->h1, B * S, HIDDEN, lw.w_up, cfg.inter, HIDDEN, ws->ffn_up,
                  B * S, cfg.inter);
  if (lw.b_up)
    add_bias_lastdim_inplace(ws->ffn_up, B * S, cfg.inter, lw.b_up, cfg.inter);

  // SwiGLU (SIMD optimized)
  simd::swiglu(ws->ffn_gate, B * S * cfg.inter, ws->ffn_up, B * S * cfg.inter,
               ws->ffn_mid, B * S * cfg.inter);

  // down projection (SIMD optimized)
  simd::matmul_nt(ws->ffn_mid, B * S, cfg.inter, lw.w_down, HIDDEN, cfg.inter,
                  ws->ffn_out, B * S, HIDDEN);
  if (lw.b_down)
    add_bias_lastdim_inplace(ws->ffn_out, B * S, HIDDEN, lw.b_down, HIDDEN);

  // residual (SIMD optimized)
  simd::add(out, B * S * HIDDEN, ws->ffn_out, B * S * HIDDEN, out, B * S * HIDDEN);
}

inline void minimind_forward_simd(const minimind_config& cfg,
                                  const minimind_weights& w,
                                  const int32_t* input_ids, int64_t B,
                                  int64_t S, const uint8_t* attention_mask_bt,
                                  int64_t m1, int64_t m0,
                                  kv_cache* cache,
                                  minimind_workspace* ws, float* out_logits,
                                  int64_t o2, int64_t o1, int64_t o0) {
  assert(o2 == B && o1 == S && o0 == cfg.vocab_size);

  // 0) embed
  embedding_lookup_bsh(w.tok_embedding, cfg.vocab_size, cfg.hidden, input_ids,
                       B, S, ws->h0, B, S, cfg.hidden);

  // 1) layer stack
  for (int64_t l = 0; l < cfg.n_layers; ++l) {
    minimind_attention_forward_simd(cfg, w.layers[l], w.rope_cos, w.rope_sin,
                                    cfg.max_pos, attention_mask_bt, m1, m0,
                                    cache, static_cast<int>(l), ws->h0, B, S,
                                    cfg.hidden, ws, ws->h1);

    minimind_ffn_forward_simd(cfg, w.layers[l], ws->h1, B, S, cfg.hidden, ws,
                              ws->h0);
  }

  // 2) final norm (SIMD optimized)
  simd::rms_norm_lastdim(ws->h0, B * S, cfg.hidden, w.final_rms, cfg.hidden,
                         cfg.rms_eps, ws->h1);

  // 3) lm_head (SIMD optimized)
  simd::matmul_nt(ws->h1, B * S, cfg.hidden, w.lm_head, cfg.vocab_size, cfg.hidden,
                  out_logits, B * S, cfg.vocab_size);

  if (cache != nullptr) cache->cur_t += S;
}

// ============================
// JSON helpers (same as baseline)
// ============================

#include <chrono>
#include <fstream>

using json = nlohmann::json;

static void write_f32(const std::string& path, const float* x, size_t n) {
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(x),
            static_cast<std::streamsize>(n * sizeof(float)));
}

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

static std::vector<float> load_tensor_from_json(const json& j) {
  std::string b64_data = j["data"];
  std::vector<uint8_t> decoded = b64_decode(b64_data);
  if (decoded.size() % sizeof(float) != 0) {
    std::cerr << "[ERROR] Decoded data size " << decoded.size()
              << " is not a multiple of sizeof(float)=" << sizeof(float) << "\n";
  }
  std::vector<float> result(decoded.size() / sizeof(float));
  std::memcpy(result.data(), decoded.data(), decoded.size());
  return result;
}

int main(int argc, char** argv) {
  std::string json_path = (argc >= 2) ? argv[1] : "dump_minimind/minimind.json";
  std::string dump_dir = (argc >= 3) ? argv[2] : "dump_minimind";
  g_dump_dir = dump_dir;

  std::cout << "========================================\n";
  std::cout << "   SIMD-Optimized (AVX2) Inference\n";
  std::cout << "========================================\n";

  // Load JSON
  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    std::cerr << "Failed to open JSON: " << json_path << "\n";
    return 1;
  }
  json j;
  json_file >> j;
  json_file.close();
  std::cout << "[OK] Loaded JSON from: " << json_path << "\n";

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
  std::cout << "Input: B=" << B << " S=" << S << "\n";

  // Load inputs
  std::string input_ids_b64 = j["inputs"]["input_ids"]["data"];
  std::vector<uint8_t> input_ids_bytes = b64_decode(input_ids_b64);
  std::vector<int32_t> input_ids(input_ids_bytes.size() / sizeof(int32_t));
  std::memcpy(input_ids.data(), input_ids_bytes.data(), input_ids_bytes.size());

  std::string attn_mask_b64 = j["inputs"]["attention_mask"]["data"];
  std::vector<uint8_t> attn_mask_bytes = b64_decode(attn_mask_b64);
  std::vector<uint8_t> attn_mask = attn_mask_bytes;

  // Load rope
  std::vector<float> rope_cos = load_tensor_from_json(j["rope"]["cos"]);
  std::vector<float> rope_sin = load_tensor_from_json(j["rope"]["sin"]);

  // Load weights
  std::vector<float> tok_embedding = load_tensor_from_json(j["weights"]["tok_embedding"]);
  std::vector<float> final_rms = load_tensor_from_json(j["weights"]["final_rms"]);
  std::vector<float> lm_head = load_tensor_from_json(j["weights"]["lm_head"]);

  // Load layers
  std::vector<minimind_layer_weights> layer_w(static_cast<size_t>(layers));
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
  std::cout << "[OK] Loaded all weights from JSON\n";

  // Build config & weights
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
  cfg.use_moe = false;

  minimind_weights W;
  W.tok_embedding = tok_embedding.data();
  W.final_rms = final_rms.data();
  W.lm_head = lm_head.data();
  W.rope_cos = rope_cos.data();
  W.rope_sin = rope_sin.data();
  W.layers = layer_w.data();

  // Workspace allocation
  const int64_t T = S;
  auto h0_data = std::make_unique<float[]>(B * S * hidden);
  auto h1_data = std::make_unique<float[]>(B * S * hidden);
  auto q_data = std::make_unique<float[]>(B * S * heads * head_dim);
  auto k_data = std::make_unique<float[]>(B * S * kv_heads * head_dim);
  auto v_data = std::make_unique<float[]>(B * S * kv_heads * head_dim);
  auto k_rep_data = std::make_unique<float[]>(B * T * heads * head_dim);
  auto v_rep_data = std::make_unique<float[]>(B * T * heads * head_dim);
  auto q_bhsd_data = std::make_unique<float[]>(B * heads * S * head_dim);
  auto k_bhtd_data = std::make_unique<float[]>(B * heads * T * head_dim);
  auto v_bhtd_data = std::make_unique<float[]>(B * heads * T * head_dim);
  auto scores_data = std::make_unique<float[]>(B * heads * S * T);
  auto probs_data = std::make_unique<float[]>(B * heads * S * T);
  auto attn_out_data = std::make_unique<float[]>(B * heads * S * head_dim);
  auto attn_out_bshd_data = std::make_unique<float[]>(B * S * heads * head_dim);
  auto attn_out_flat_data = std::make_unique<float[]>(B * S * heads * head_dim);
  auto ffn_gate_data = std::make_unique<float[]>(B * S * inter);
  auto ffn_up_data = std::make_unique<float[]>(B * S * inter);
  auto ffn_mid_data = std::make_unique<float[]>(B * S * inter);
  auto ffn_out_data = std::make_unique<float[]>(B * S * hidden);
  auto logits_data = std::make_unique<float[]>(B * S * vocab);

  std::fill_n(h0_data.get(), B * S * hidden, 0.0f);
  std::fill_n(h1_data.get(), B * S * hidden, 0.0f);
  std::fill_n(logits_data.get(), B * S * vocab, 0.0f);

  minimind_workspace ws;
  ws.h0 = h0_data.get();
  ws.h1 = h1_data.get();
  ws.q = q_data.get();
  ws.k = k_data.get();
  ws.v = v_data.get();
  ws.k_rep = k_rep_data.get();
  ws.v_rep = v_rep_data.get();
  ws.q_bhsd = q_bhsd_data.get();
  ws.k_bhtd = k_bhtd_data.get();
  ws.v_bhtd = v_bhtd_data.get();
  ws.scores = scores_data.get();
  ws.probs = probs_data.get();
  ws.attn_out = attn_out_data.get();
  ws.attn_out_bshd = attn_out_bshd_data.get();
  ws.attn_out_flat = attn_out_flat_data.get();
  ws.ffn_gate = ffn_gate_data.get();
  ws.ffn_up = ffn_up_data.get();
  ws.ffn_mid = ffn_mid_data.get();
  ws.ffn_out = ffn_out_data.get();
  ws.logits = logits_data.get();

  // Run forward (timed)
  kv_cache* cache = nullptr;

  auto start = std::chrono::high_resolution_clock::now();
  minimind_forward_simd(cfg, W, input_ids.data(), B, S, attn_mask.data(), B, T,
                        cache, &ws, logits_data.get(), B, S, vocab);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "\n========================================\n";
  std::cout << "[Forward] Shape: (" << B << ", " << S << ", " << vocab << ")\n";
  std::cout << "[Timing] SIMD Forward pass: " << elapsed_ms << "ms\n";

  // Compute logits stats
  float min_logit = logits_data[0];
  float max_logit = logits_data[0];
  double mean_logit = 0.0;
  for (int64_t i = 0; i < B * S * vocab; ++i) {
    min_logit = std::min(min_logit, logits_data[i]);
    max_logit = std::max(max_logit, logits_data[i]);
    mean_logit += logits_data[i];
  }
  mean_logit /= (B * S * vocab);

  std::cout << "[Logits] Min: " << min_logit << ", Max: " << max_logit
            << ", Mean: " << mean_logit << "\n";
  std::cout << "========================================\n\n";

  // Save logits
  std::string logits_path = dump_dir + "/logits_simd.npy";
  write_f32(logits_path, logits_data.get(), B * S * vocab);
  std::cout << "[OK] Saved logits to: " << logits_path << "\n";
  std::cout << "[OK] SIMD Inference complete\n";

  return 0;
}

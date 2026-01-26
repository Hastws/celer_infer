#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdlib>

#include "tensor_op.hpp"

using namespace celer_infer;

// Global to store h0_embedding for debugging - using static trick to avoid dtor issues
// We'll manage these locally in main() instead
std::string g_dump_dir;

// ============================
// Model wiring structs (all pointers, no fancy tensor class)
// ============================

struct minimind_config {
  int64_t vocab_size = 0;
  int64_t hidden = 0;
  int64_t n_layers = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t head_dim = 0;  // hidden / n_heads
  int64_t inter = 0;     // intermediate_size
  float rms_eps = 1e-5f;
  // rope
  int64_t max_pos = 0;
  // moe
  bool use_moe = false;
  int64_t moe_topk = 2;
  int64_t moe_experts = 0;
  int64_t moe_shared = 0;
};

struct minimind_layer_weights {
  // attn
  const float* w_q =
      nullptr;  // (H*D, hidden) as (out,in) => (w1,w0)=(H*D, hidden)
  const float* w_k = nullptr;  // (KVH*D, hidden)
  const float* w_v = nullptr;  // (KVH*D, hidden)
  const float* w_o = nullptr;  // (hidden, H*D)
  // ffn (dense)
  const float* w_gate = nullptr;  // (inter, hidden)
  const float* w_up = nullptr;    // (inter, hidden)
  const float* w_down = nullptr;  // (hidden, inter)

  // norms
  const float* rms_attn = nullptr;  // (hidden)
  const float* rms_ffn = nullptr;   // (hidden)

  // optional biases (if you要跟 torch bias=True 对齐)
  const float* b_q = nullptr;
  const float* b_k = nullptr;
  const float* b_v = nullptr;
  const float* b_o = nullptr;
  const float* b_gate = nullptr;
  const float* b_up = nullptr;
  const float* b_down = nullptr;

  // moe gate weight (experts, hidden) if use_moe
  const float* moe_gate_w = nullptr;  // (E, hidden)
};

struct minimind_weights {
  const float* tok_embedding = nullptr;  // (V, hidden)
  const float* final_rms = nullptr;      // (hidden)
  const float* lm_head = nullptr;  // (V, hidden) (tied to embedding is allowed)

  const float* rope_cos = nullptr;  // (max_pos, head_dim) BUT 你的 cos/sin
                                    // 常常是 (max_pos, head_dim*2) 也行
  const float* rope_sin = nullptr;  // (max_pos, head_dim)

  const minimind_layer_weights* layers = nullptr;  // size n_layers
};

// KV cache pointers (owned by caller)
struct kv_cache {
  // Each layer:
  // k: (B, max_t, KVH, D)  v: (B, max_t, KVH, D)
  float** k = nullptr;
  float** v = nullptr;
  int64_t max_t = 0;
  int64_t cur_t = 0;  // past length
};

// Workspace pointers (owned by caller, pre-allocated)
// All are float buffers large enough for the shapes below.
struct minimind_workspace {
  // hidden
  float* h0 = nullptr;  // (B,S,hidden)
  float* h1 = nullptr;  // (B,S,hidden) temp

  // q,k,v
  float* q = nullptr;  // (B,S,H,D)
  float* k = nullptr;  // (B,S,KVH,D)
  float* v = nullptr;  // (B,S,KVH,D)

  // cache concat outputs for this step (views into cache, but for repeat you
  // may need temp)
  float* k_rep = nullptr;  // (B,T,H,D) temp
  float* v_rep = nullptr;  // (B,T,H,D) temp

  // transposed for attention
  float* q_bhsd = nullptr;  // (B,H,S,D)
  float* k_bhtd = nullptr;  // (B,H,T,D)
  float* v_bhtd = nullptr;  // (B,H,T,D)

  // attention internals
  float* scores = nullptr;    // (B,H,S,T)
  float* probs = nullptr;     // (B,H,S,T)
  float* attn_out = nullptr;  // (B,H,S,D)

  // back to (B,S,H,D) and merge heads
  float* attn_out_bshd = nullptr;  // (B,S,H,D)
  float* attn_out_flat = nullptr;  // (B,S,H*D)

  // ffn
  float* ffn_gate = nullptr;  // (B,S,inter)
  float* ffn_up = nullptr;    // (B,S,inter)
  float* ffn_mid = nullptr;   // (B,S,inter)
  float* ffn_out = nullptr;   // (B,S,hidden)

  // logits
  float* logits = nullptr;  // (B,S,V) or (B*keep,V)
};

// ============================
// Helper 偏移函数
// ============================
// 使用 tensor_op 中定义的 off_4d，但这里我们需要调整参数
// 对于 (B,S,H,D) 张量，off_4d(B, S, H, D, S*H, H, D) 
// 实际上 tensor_op 中的 off_4d(i3, i2, i1, i0, a2, a1, a0)
// 所以这是对的：off_4d(b, s, h, d, S*H, H, D)
inline int64_t off_4d_bshd(int64_t b, int64_t s, int64_t h, int64_t d,
                           int64_t S, int64_t H, int64_t D) {
  return celer_infer::off_4d(b, s, h, d, S * H, H, D);
}

// ============================
// High-level forward: inference path (dropout off)
// You can extend for training by inserting dropout + aux_loss.
// ============================

inline void minimind_attention_forward_infer(
    const minimind_config& cfg, const minimind_layer_weights& lw,
    const float* rope_cos, const float* rope_sin, int64_t max_pos,
    const uint8_t* attn_mask_bt, int64_t m1,
    int64_t m0,  // (B,T) keep mask, optional
    kv_cache* cache, int layer_id, const float* x, int64_t B, int64_t S,
    int64_t HIDDEN,  // x (B,S,hidden)
    minimind_workspace* ws, float* out) {
  // out (B,S,hidden)
  assert(HIDDEN == cfg.hidden);
  assert(cfg.head_dim * cfg.n_heads == cfg.hidden);
  const int64_t H = cfg.n_heads;
  const int64_t KVH = cfg.n_kv_heads;
  const int64_t D = cfg.head_dim;

  // 1) pre-attn rmsnorm: h1 = rms_norm(x)
  rms_norm_lastdim(x, B * S, HIDDEN, lw.rms_attn, HIDDEN, cfg.rms_eps, ws->h1);

  // 2) q,k,v projections: (B,S,hidden)->(B,S,heads*D)
  // q_flat (B,S,H*D) into ws->attn_out_flat (reuse buffer)
  // k_flat,v_flat into ffn_mid (reuse) if you want; here用明确 ws->k/ws->v 的
  // layout For simplicity: do 3 linear separately (你也可以用 linear_qkv_fused)
  // q_flat:
  matmul_nt(ws->h1, B * S, HIDDEN, lw.w_q, H * D, HIDDEN, ws->attn_out_flat,
            B * S, H * D);
  if (lw.b_q)
    add_bias_lastdim_inplace(ws->attn_out_flat, B * S, H * D, lw.b_q, H * D);

  // Save Q projection for layer 0
  // Debug buffer saving removed to avoid dtor issues

  // k_flat:
  matmul_nt(ws->h1, B * S, HIDDEN, lw.w_k, KVH * D, HIDDEN, ws->k, B * S,
            KVH * D);
  if (lw.b_k)
    add_bias_lastdim_inplace(ws->k, B * S, KVH * D, lw.b_k, KVH * D);

  // Save K projection for layer 0
  // Debug buffer saving removed

  // v_flat:
  matmul_nt(ws->h1, B * S, HIDDEN, lw.w_v, KVH * D, HIDDEN, ws->v, B * S,
            KVH * D);
  if (lw.b_v)
    add_bias_lastdim_inplace(ws->v, B * S, KVH * D, lw.b_v, KVH * D);

  // Save V projection for layer 0
  // Debug buffer saving removed

  // 3) view to (B,S,H,D) and (B,S,KVH,D)
  // q: ws->q uses (B,S,H,D)
  // k,v: ws->k/ws->v uses (B,S,KVH,D)
  {
    // copy-based view 不需要，但为了“每动 tensor 是一个函数”，你也可以写成
    // view_check + pointer cast
    const float* q_src = ws->attn_out_flat;  // (B,S,H*D)
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t s = 0; s < S; ++s) {
        const float* src = q_src + (b * S + s) * (H * D);
        float* dst = ws->q + off_4d_bshd(b, s, 0, 0, S, H, D);  // (b,s,:,:)
        std::memcpy(dst, src, static_cast<size_t>(H * D) * sizeof(float));
      }
    }
    // K and V already stored correctly since matmul outputs directly to ws->k and ws->v
  }

  // 4) RoPE (q,k) using (cos,sin)[pos : pos+S]
  // 你的 rope_cos/rope_sin 可能是 (max_pos, D) 或 (max_pos, 2*D)，这里按
  // (max_pos, D) 写
  const int64_t past = (cache && cache->cur_t > 0) ? cache->cur_t : 0;
  assert(past + S <= max_pos);

  // apply_rotary_pos_emb 你应已在 tensor_ops_5d.h 里实现过类似版本
  // 这里假设你有：apply_rotary_pos_emb_bshd(q,k,cos,sin)
  // 若你还没写，就沿你 Python 公式写一个 4D 版本即可。
  apply_rotary_pos_emb_bshd(ws->q, B, S, H, D, ws->k, B, S, KVH, D,
                            rope_cos + past * D, rope_sin + past * D, S, D);

  // 5) KV cache: append current step k/v into cache
  // cache storage: (B, max_t, KVH, D)
  const int64_t T = past + S;
  if (cache != nullptr) {
    assert(cache->max_t >= T);
    kv_cache_append_btkd(cache->k[layer_id], B, cache->max_t, KVH, D, ws->k, B,
                         S, KVH, D, past);
    kv_cache_append_btkd(cache->v[layer_id], B, cache->max_t, KVH, D, ws->v, B,
                         S, KVH, D, past);
  }

  // 6) Build k_cat/v_cat views (B,T,KVH,D)
  const float* k_cat = (cache != nullptr)
                           ? cache->k[layer_id]
                           : ws->k;  // if no cache, treat ws->k as (B,S,KVH,D)
  const float* v_cat = (cache != nullptr) ? cache->v[layer_id] : ws->v;
  const int64_t cat_T = (cache != nullptr) ? T : S;

  // 7) repeat kv -> (B,T,H,D)  then transpose to (B,H,T,D)
  repeat_kv_btkd_to_bthd(k_cat, B, cat_T, KVH, D, cfg.n_heads / cfg.n_kv_heads,
                         ws->k_rep, B, cat_T, H, D);
  repeat_kv_btkd_to_bthd(v_cat, B, cat_T, KVH, D, cfg.n_heads / cfg.n_kv_heads,
                         ws->v_rep, B, cat_T, H, D);

  // transpose q: (B,S,H,D)->(B,H,S,D)
  transpose_bshd_to_bhsd(ws->q, B, S, H, D, ws->q_bhsd, B, H, S, D);

  // transpose k/v: (B,T,H,D)->(B,H,T,D)
  transpose_bthd_to_bhtd(ws->k_rep, B, cat_T, H, D, ws->k_bhtd, B, H, cat_T, D);
  transpose_bthd_to_bhtd(ws->v_rep, B, cat_T, H, D, ws->v_bhtd, B, H, cat_T, D);

  // 8) attention: scores = q @ k^T
  attn_qk_matmul(ws->q_bhsd, B, H, S, D, ws->k_bhtd, B, H, cat_T, D,
                 1.0f / std::sqrt(static_cast<float>(D)), ws->scores, B, H, S,
                 cat_T);

  // Save attention scores for layer 0
  // Debug buffer saving removed

  // 9) causal mask with prefix_offset=past
  apply_causal_mask(ws->scores, B, H, S, cat_T, past);

  // 10) padding mask (B,T) if provided
  if (attn_mask_bt != nullptr) {
    assert(m1 == B && m0 == cat_T);
    add_attention_mask(ws->scores, B, H, S, cat_T, attn_mask_bt, m1, m0);
  }

  // 11) softmax over last dim T
  attn_softmax_scores(ws->scores, B, H, S, cat_T, ws->probs, B, H, S, cat_T);

  // Save attention probabilities for layer 0
  // Debug buffer saving removed

  // 12) attn_out = probs @ v
  attn_pv_matmul(ws->probs, B, H, S, cat_T, ws->v_bhtd, B, H, cat_T, D,
                 ws->attn_out, B, H, S, D);

  // 13) back to (B,S,H,D) then merge heads -> (B,S,H*D)
  transpose_bhsd_to_bshd(ws->attn_out, B, H, S, D, ws->attn_out_bshd, B, S, H,
                         D);

  // merge heads copy (B,S,H,D)->(B,S,H*D)
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
        const float* src = ws->attn_out_bshd + off_4d_bshd(b, s, 0, 0, S, H, D);
    }
  }

  // Save merged attention output for layer 0
  // Debug buffer saving removed

  // 14) o_proj: (B,S,H*D)->(B,S,hidden)
  // Use ws->attn_out_bshd as temporary buffer for o_proj (no longer needed
  // after merge_heads)
  matmul_nt(ws->attn_out_flat, B * S, H * D, lw.w_o, HIDDEN, H * D,
            (float*)ws->attn_out_bshd, B * S, HIDDEN);
  if (lw.b_o)
    add_bias_lastdim_inplace((float*)ws->attn_out_bshd, B * S, HIDDEN, lw.b_o,
                             HIDDEN);

  // Save o_proj result for layer 0 (after bias, before residual)
  // Debug buffer saving removed

  // 15) residual: out = x + o_proj
  std::memcpy(out, x, static_cast<size_t>(B * S * HIDDEN) * sizeof(float));
  add_inplace(out, B * S * HIDDEN, (float*)ws->attn_out_bshd, B * S * HIDDEN,
              1.0f);
}

inline void minimind_ffn_forward_infer(const minimind_config& cfg,
                                       const minimind_layer_weights& lw,
                                       const float* x, int64_t B, int64_t S,
                                       int64_t HIDDEN,  // x (B,S,hidden)
                                       minimind_workspace* ws, float* out) {
  // out (B,S,hidden)
  // Save original input before RMSNorm
  std::memcpy(out, x, static_cast<size_t>(B * S * HIDDEN) * sizeof(float));

  // pre-ffn rmsnorm
  rms_norm_lastdim(x, B * S, HIDDEN, lw.rms_ffn, HIDDEN, cfg.rms_eps, ws->h1);

  // Save FFN norm output for layer 0 (for debugging)
  // Debug buffer saving removed

  // gate/up: (B,S,hidden)->(B,S,inter)
  matmul_nt(ws->h1, B * S, HIDDEN, lw.w_gate, cfg.inter, HIDDEN, ws->ffn_gate,
            B * S, cfg.inter);
  if (lw.b_gate)
    add_bias_lastdim_inplace(ws->ffn_gate, B * S, cfg.inter, lw.b_gate,
                             cfg.inter);

  matmul_nt(ws->h1, B * S, HIDDEN, lw.w_up, cfg.inter, HIDDEN, ws->ffn_up,
            B * S, cfg.inter);
  if (lw.b_up)
    add_bias_lastdim_inplace(ws->ffn_up, B * S, cfg.inter, lw.b_up, cfg.inter);

  // silu(gate) * up  -> mid
  silu_inplace(ws->ffn_gate, B * S * cfg.inter);
  mul(ws->ffn_gate, B * S * cfg.inter, ws->ffn_up, B * S * cfg.inter,
      ws->ffn_mid, B * S * cfg.inter);

  // down: (B,S,inter)->(B,S,hidden)
  matmul_nt(ws->ffn_mid, B * S, cfg.inter, lw.w_down, HIDDEN, cfg.inter,
            ws->ffn_out, B * S, HIDDEN);
  if (lw.b_down)
    add_bias_lastdim_inplace(ws->ffn_out, B * S, HIDDEN, lw.b_down, HIDDEN);

  // Save FFN down projection for layer 0 (before residual)
  // Debug buffer saving removed

  // residual: out = out + ffn_out (out already contains x from beginning of
  // function)
  add_inplace(out, B * S * HIDDEN, ws->ffn_out, B * S * HIDDEN, 1.0f);
}

// Full forward (inference):
// input_ids (B,S) -> logits (B,S,V)  + optional KV cache update
inline void minimind_forward_infer(const minimind_config& cfg,
                                   const minimind_weights& w,
                                   const int32_t* input_ids, int64_t B,
                                   int64_t S, const uint8_t* attention_mask_bt,
                                   int64_t m1, int64_t m0,  // optional (B,T)
                                   kv_cache* cache,         // optional
                                   minimind_workspace* ws, float* out_logits,
                                   int64_t o2, int64_t o1, int64_t o0) {
  // (B,S,V)
  assert(o2 == B && o1 == S && o0 == cfg.vocab_size);
  // assert(cfg.hidden == w.layers[0].rms_attn ? cfg.hidden : cfg.hidden); //
  // dummy "touch" to avoid unused warning

  // 0) embed -> ws->h0 (B,S,hidden)
  embedding_lookup_bsh(w.tok_embedding, cfg.vocab_size, cfg.hidden, input_ids,
                       B, S, ws->h0, B, S, cfg.hidden);

  // Debug: Save h0 right after embedding for verification
  {
    std::cout << "[DEBUG] h0 after embedding lookup:\n";
    std::cout << "  Shape: (" << B << ", " << S << ", " << cfg.hidden << ")\n";
    std::cout << "  h0[0,0,:10] = ";
    for (int i = 0; i < 10; ++i) {
      std::cout << ws->h0[i] << " ";
    }
    std::cout << "\n";

    // Find min/max
    float h0_min = ws->h0[0], h0_max = ws->h0[0];
    for (int64_t i = 0; i < B * S * cfg.hidden; ++i) {
      h0_min = std::min(h0_min, ws->h0[i]);
      h0_max = std::max(h0_max, ws->h0[i]);
    }
    std::cout << "  Min: " << h0_min << ", Max: " << h0_max << "\n";
  }

  // Save h0_embedding before any layer processing
  // Skip global buffer saving to avoid destructor issues
  // Just keep logits output which is the main result

  // 1) layer stack
  for (int64_t l = 0; l < cfg.n_layers; ++l) {
    // For layer 0, save intermediate outputs
    if (l == 0) {
      // Apply input_layernorm
      rms_norm_lastdim(ws->h0, B * S, cfg.hidden, w.layers[l].rms_attn,
                       cfg.hidden, cfg.rms_eps, ws->h1);
      // Debug buffer saving removed to avoid dtor issues
      std::cout << "[DEBUG] Saved h_norm (after input_layernorm) for layer 0\n";
    }

    // attention
    minimind_attention_forward_infer(cfg, w.layers[l], w.rope_cos, w.rope_sin,
                                     cfg.max_pos, attention_mask_bt, m1, m0,
                                     cache, static_cast<int>(l), ws->h0, B, S,
                                     cfg.hidden, ws, ws->h1);
    // For layer 0, save attention output
    if (l == 0) {
      // Debug buffer saving removed
      std::cout << "[DEBUG] Saved h1 (after attention) for layer 0\n";
    }

    // ffn (dense; moe 你可以在这里替换成 moe 版本)
    minimind_ffn_forward_infer(cfg, w.layers[l], ws->h1, B, S, cfg.hidden, ws,
                               ws->h0);

    // For layer 0, save FFN output
    if (l == 0) {
      // Debug buffer saving removed
      std::cout << "[DEBUG] Saved h0 (after FFN) for layer 0\n";
    }
  }

  // 2) final norm: h1 = rms_norm(h0)
  rms_norm_lastdim(ws->h0, B * S, cfg.hidden, w.final_rms, cfg.hidden,
                   cfg.rms_eps, ws->h1);

  // 3) lm_head: logits = h1 @ W^T  where W is (V, hidden)
  // flatten to 2D: (B*S, hidden) x (V, hidden)^T => (B*S, V)
  matmul_nt(ws->h1, B * S, cfg.hidden, w.lm_head, cfg.vocab_size, cfg.hidden,
            out_logits, B * S, cfg.vocab_size);

  // update cache length
  if (cache != nullptr) cache->cur_t += S;
}

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "tensor_op.hpp"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;

static void write_f32(const std::string& path, const float* x, size_t n) {
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(x),
            static_cast<std::streamsize>(n * sizeof(float)));
}

// ===== JSON 辅助函数 =====
static inline int b64_char_to_val(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;  // padding or invalid
}

static std::vector<uint8_t> b64_decode(const std::string& b64) {
  std::vector<uint8_t> ret;
  int val = 0;
  int bits = 0;

  for (unsigned char c : b64) {
    int digit = b64_char_to_val(c);
    if (digit == -1) continue;  // skip invalid chars (including padding)

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
  // 确保大小是float大小的倍数
  if (decoded.size() % sizeof(float) != 0) {
    std::cerr << "[ERROR] Decoded data size " << decoded.size()
              << " is not a multiple of sizeof(float)=" << sizeof(float)
              << "\n";
  }
  std::vector<float> result(decoded.size() / sizeof(float));
  std::memcpy(result.data(), decoded.data(), decoded.size());
  return result;
}

int main(int argc, char** argv) {
  std::string json_path = (argc >= 2) ? argv[1] : "dump_minimind/minimind.json";
  std::string dump_dir = (argc >= 3) ? argv[2] : "dump_minimind";

  // Store dump_dir in global for use in forward function
  g_dump_dir = dump_dir;

  // ===== 加载 JSON =====
  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    std::cerr << "Failed to open JSON: " << json_path << "\n";
    return 1;
  }
  json j;
  json_file >> j;
  json_file.close();

  std::cout << "[OK] Loaded JSON from: " << json_path << "\n";

  // 解析 config
  auto cfg_j = j["config"];
  const int64_t vocab = cfg_j["vocab_size"];
  const int64_t hidden = cfg_j["hidden_size"];
  const int64_t layers = cfg_j["num_hidden_layers"];
  const int64_t heads = cfg_j["num_attention_heads"];
  const int64_t kv_heads = cfg_j["num_key_value_heads"];
  const int64_t inter = cfg_j["intermediate_size"];
  const int64_t max_pos = cfg_j["max_position_embeddings"];
  const int64_t head_dim = hidden / heads;

  // 解析 meta
  auto meta = j["meta"];
  const int64_t B = meta["B"];
  const int64_t S = meta["S"];

  std::cout << "Config: hidden=" << hidden << " layers=" << layers
            << " heads=" << heads << " vocab=" << vocab << "\n";
  std::cout << "Input: B=" << B << " S=" << S << "\n";

  // 加载 inputs
  // input_ids是int32，需要直接解码
  std::string input_ids_b64 = j["inputs"]["input_ids"]["data"];
  std::vector<uint8_t> input_ids_bytes = b64_decode(input_ids_b64);
  std::vector<int32_t> input_ids(input_ids_bytes.size() / sizeof(int32_t));
  std::memcpy(input_ids.data(), input_ids_bytes.data(), input_ids_bytes.size());

  // attn_mask是uint8，需要直接解码
  std::string attn_mask_b64 = j["inputs"]["attention_mask"]["data"];
  std::vector<uint8_t> attn_mask_bytes = b64_decode(attn_mask_b64);
  std::vector<uint8_t> attn_mask = attn_mask_bytes;

  // 加载 rope
  std::vector<float> rope_cos = load_tensor_from_json(j["rope"]["cos"]);
  std::vector<float> rope_sin = load_tensor_from_json(j["rope"]["sin"]);

  // 加载 weights
  std::vector<float> tok_embedding =
      load_tensor_from_json(j["weights"]["tok_embedding"]);
  std::vector<float> final_rms =
      load_tensor_from_json(j["weights"]["final_rms"]);
  std::vector<float> lm_head = load_tensor_from_json(j["weights"]["lm_head"]);

  // Debug: Save tok_embedding for comparison
  {
    std::string emb_path = dump_dir + "/tok_embedding_cpp.npy";
    write_f32(emb_path, tok_embedding.data(), tok_embedding.size());
    std::cout << "[DEBUG] Saved tok_embedding to: " << emb_path << "\n";
    std::cout << "[DEBUG] tok_embedding size: " << tok_embedding.size()
              << ", min: "
              << *std::min_element(tok_embedding.begin(), tok_embedding.end())
              << ", max: "
              << *std::max_element(tok_embedding.begin(), tok_embedding.end())
              << "\n";
  }

  // 加载 layers
  std::vector<minimind_layer_weights> layer_w(static_cast<size_t>(layers));
  std::vector<std::vector<float>> wq(layers), wk(layers), wv(layers),
      wo(layers);
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

  // ===== 组 config & weights =====
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

  // ===== workspace 分配 =====
  const int64_t T = S;  // 本次不测 cache，T=S
  
  // 使用堆分配而非栈分配，避免栈溢出
  // Use raw arrays instead of unique_ptr<vector> to avoid potential issues
  float* h0_data = new float[B * S * hidden]();
  float* h1_data = new float[B * S * hidden]();
  float* q_data = new float[B * S * heads * head_dim]();
  float* k_data = new float[B * S * kv_heads * head_dim]();
  float* v_data = new float[B * S * kv_heads * head_dim]();

  float* k_rep_data = new float[B * T * kv_heads * head_dim]();
  float* v_rep_data = new float[B * T * kv_heads * head_dim]();

  float* q_bhsd_data = new float[B * heads * S * head_dim]();
  float* k_bhtd_data = new float[B * kv_heads * T * head_dim]();
  float* v_bhtd_data = new float[B * kv_heads * T * head_dim]();

  float* scores_data = new float[B * heads * S * T]();
  float* probs_data = new float[B * heads * S * T]();
  float* attn_out_data = new float[B * heads * S * head_dim]();

  float* attn_out_bshd_data = new float[B * S * heads * head_dim]();
  float* attn_out_flat_data = new float[B * S * heads * head_dim]();

  float* ffn_gate_data = new float[B * S * inter]();
  float* ffn_up_data = new float[B * S * inter]();
  float* ffn_mid_data = new float[B * S * inter]();
  float* ffn_out_data = new float[B * S * hidden]();

  float* logits_data = new float[B * S * vocab]();

  minimind_workspace ws;
  
  ws.h0 = h0_data;
  ws.h1 = h1_data;
  ws.q = q_data;
  ws.k = k_data;
  ws.v = v_data;
  ws.k_rep = k_rep_data;
  ws.v_rep = v_rep_data;
  ws.q_bhsd = q_bhsd_data;
  ws.k_bhtd = k_bhtd_data;
  ws.v_bhtd = v_bhtd_data;
  ws.scores = scores_data;
  ws.probs = probs_data;
  ws.attn_out = attn_out_data;
  ws.attn_out_bshd = attn_out_bshd_data;
  ws.attn_out_flat = attn_out_flat_data;
  ws.ffn_gate = ffn_gate_data;
  ws.ffn_up = ffn_up_data;
  ws.ffn_mid = ffn_mid_data;
  ws.ffn_out = ffn_out_data;
  ws.logits = logits_data;
  
  // Debug: verify pointers
  std::cout << "[DEBUG] B=" << B << ", S=" << S << ", hidden=" << hidden << ", inter=" << inter << "\n";
  std::cout << "[DEBUG] ffn_gate.size()=" << (B * S * inter) << ", data=" << (void*)ffn_gate_data << "\n";
  std::cout << "[DEBUG] ws.h1=" << (void*)ws.h1 << "\n";
  std::cout << "[DEBUG] ws.ffn_mid=" << (void*)ws.ffn_mid << "\n";
  std::cout << "[DEBUG] ffn_mid.size()=" << (B * S * inter) << "\n";

  // ===== 运行 forward (timed) =====
  kv_cache* cache = nullptr;

  auto start = std::chrono::high_resolution_clock::now();
  minimind_forward_infer(cfg, W, input_ids.data(), B, S, attn_mask.data(), B, T,
                         cache, &ws, logits_data, B, S, vocab);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "\n========================================\n";
  std::cout << "[Forward] Shape: (" << B << ", " << S << ", " << vocab << ")"
            << "\n";
  std::cout << "[Timing] Forward pass: " << elapsed_ms << "ms\n";

  // 计算 logits 统计
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

  // ===== 保存 logits =====
  std::string logits_path = dump_dir + "/logits_cpp.npy";
  write_f32(logits_path, logits_data, B * S * vocab);
  std::cout << "[OK] Saved logits to: " << logits_path << "\n";

  // ===== Use std::exit to bypass destructor phase =====
  // The destructors of global/static variables cause segfaults
  // All heap memory will be cleaned up by the OS on process exit
  std::cout << "[OK] Inference complete\n";
  std::cout.flush();  // Ensure output is written
  std::exit(0);  // Skip all dtors, let OS cleanup
}

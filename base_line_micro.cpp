#include <iostream>
#include <vector>
#include <algorithm>

#include "tensor_op.hpp"

using namespace celer_infer;

// ============================
// Model wiring structs (all pointers, no fancy tensor class)
// ============================

struct minimind_config {
    int64_t vocab_size = 0;
    int64_t hidden = 0;
    int64_t n_layers = 0;
    int64_t n_heads = 0;
    int64_t n_kv_heads = 0;
    int64_t head_dim = 0; // hidden / n_heads
    int64_t inter = 0; // intermediate_size
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
    const float *w_q = nullptr; // (H*D, hidden) as (out,in) => (w1,w0)=(H*D, hidden)
    const float *w_k = nullptr; // (KVH*D, hidden)
    const float *w_v = nullptr; // (KVH*D, hidden)
    const float *w_o = nullptr; // (hidden, H*D)
    // ffn (dense)
    const float *w_gate = nullptr; // (inter, hidden)
    const float *w_up = nullptr; // (inter, hidden)
    const float *w_down = nullptr; // (hidden, inter)

    // norms
    const float *rms_attn = nullptr; // (hidden)
    const float *rms_ffn = nullptr; // (hidden)

    // optional biases (if you要跟 torch bias=True 对齐)
    const float *b_q = nullptr;
    const float *b_k = nullptr;
    const float *b_v = nullptr;
    const float *b_o = nullptr;
    const float *b_gate = nullptr;
    const float *b_up = nullptr;
    const float *b_down = nullptr;

    // moe gate weight (experts, hidden) if use_moe
    const float *moe_gate_w = nullptr; // (E, hidden)
};

struct minimind_weights {
    const float *tok_embedding = nullptr; // (V, hidden)
    const float *final_rms = nullptr; // (hidden)
    const float *lm_head = nullptr; // (V, hidden) (tied to embedding is allowed)

    const float *rope_cos = nullptr; // (max_pos, head_dim) BUT 你的 cos/sin 常常是 (max_pos, head_dim*2) 也行
    const float *rope_sin = nullptr; // (max_pos, head_dim)

    const minimind_layer_weights *layers = nullptr; // size n_layers
};

// KV cache pointers (owned by caller)
struct kv_cache {
    // Each layer:
    // k: (B, max_t, KVH, D)  v: (B, max_t, KVH, D)
    float **k = nullptr;
    float **v = nullptr;
    int64_t max_t = 0;
    int64_t cur_t = 0; // past length
};

// Workspace pointers (owned by caller, pre-allocated)
// All are float buffers large enough for the shapes below.
struct minimind_workspace {
    // hidden
    float *h0 = nullptr; // (B,S,hidden)
    float *h1 = nullptr; // (B,S,hidden) temp

    // q,k,v
    float *q = nullptr; // (B,S,H,D)
    float *k = nullptr; // (B,S,KVH,D)
    float *v = nullptr; // (B,S,KVH,D)

    // cache concat outputs for this step (views into cache, but for repeat you may need temp)
    float *k_rep = nullptr; // (B,T,H,D) temp
    float *v_rep = nullptr; // (B,T,H,D) temp

    // transposed for attention
    float *q_bhsd = nullptr; // (B,H,S,D)
    float *k_bhtd = nullptr; // (B,H,T,D)
    float *v_bhtd = nullptr; // (B,H,T,D)

    // attention internals
    float *scores = nullptr; // (B,H,S,T)
    float *probs = nullptr; // (B,H,S,T)
    float *attn_out = nullptr; // (B,H,S,D)

    // back to (B,S,H,D) and merge heads
    float *attn_out_bshd = nullptr; // (B,S,H,D)
    float *attn_out_flat = nullptr; // (B,S,H*D)

    // ffn
    float *ffn_gate = nullptr; // (B,S,inter)
    float *ffn_up = nullptr; // (B,S,inter)
    float *ffn_mid = nullptr; // (B,S,inter)
    float *ffn_out = nullptr; // (B,S,hidden)

    // logits
    float *logits = nullptr; // (B,S,V) or (B*keep,V)
};

// ============================
// High-level forward: inference path (dropout off)
// You can extend for training by inserting dropout + aux_loss.
// ============================

inline void minimind_attention_forward_infer(
    const minimind_config &cfg,
    const minimind_layer_weights &lw,
    const float *rope_cos, const float *rope_sin, int64_t max_pos,
    const uint8_t *attn_mask_bt, int64_t m1, int64_t m0, // (B,T) keep mask, optional
    kv_cache *cache,
    int layer_id,
    const float *x, int64_t B, int64_t S, int64_t HIDDEN, // x (B,S,hidden)
    minimind_workspace *ws,
    float *out) {
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
    // k_flat,v_flat into ffn_mid (reuse) if you want; here用明确 ws->k/ws->v 的 layout
    // For simplicity: do 3 linear separately (你也可以用 linear_qkv_fused)
    // q_flat:
    matmul_nt(ws->h1, B * S, HIDDEN, lw.w_q, H * D, HIDDEN, ws->attn_out_flat, B * S, H * D);
    if (lw.b_q) add_bias_lastdim_inplace(ws->attn_out_flat, B * S, H * D, lw.b_q, H * D);

    // k_flat:
    matmul_nt(ws->h1, B * S, HIDDEN, lw.w_k, KVH * D, HIDDEN, ws->ffn_mid, B * S, KVH * D);
    if (lw.b_k) add_bias_lastdim_inplace(ws->ffn_mid, B * S, KVH * D, lw.b_k, KVH * D);

    // v_flat:
    matmul_nt(ws->h1, B * S, HIDDEN, lw.w_v, KVH * D, HIDDEN, ws->ffn_up, B * S, KVH * D);
    if (lw.b_v) add_bias_lastdim_inplace(ws->ffn_up, B * S, KVH * D, lw.b_v, KVH * D);

    // 3) view to (B,S,H,D) and (B,S,KVH,D)
    // q: ws->q uses (B,S,H,D)
    // k,v: ws->k/ws->v uses (B,S,KVH,D)
    {
        // copy-based view 不需要，但为了“每动 tensor 是一个函数”，你也可以写成 view_check + pointer cast
        const float *q_src = ws->attn_out_flat; // (B,S,H*D)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s < S; ++s) {
                const float *src = q_src + (b * S + s) * (H * D);
                float *dst = ws->q + off_4d(b, s, 0, 0, S, H, D); // (b,s,:,:)
                std::memcpy(dst, src, static_cast<size_t>(H * D) * sizeof(float));
            }
        }
        const float *k_src = ws->ffn_mid; // (B,S,KVH*D)
        const float *v_src = ws->ffn_up; // (B,S,KVH*D)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s < S; ++s) {
                std::memcpy(ws->k + off_4d(b, s, 0, 0, S, KVH, D),
                            k_src + (b * S + s) * (KVH * D),
                            static_cast<size_t>(KVH * D) * sizeof(float));
                std::memcpy(ws->v + off_4d(b, s, 0, 0, S, KVH, D),
                            v_src + (b * S + s) * (KVH * D),
                            static_cast<size_t>(KVH * D) * sizeof(float));
            }
        }
    }

    // 4) RoPE (q,k) using (cos,sin)[pos : pos+S]
    // 你的 rope_cos/rope_sin 可能是 (max_pos, D) 或 (max_pos, 2*D)，这里按 (max_pos, D) 写
    const int64_t past = (cache && cache->cur_t > 0) ? cache->cur_t : 0;
    assert(past + S <= max_pos);

    // apply_rotary_pos_emb 你应已在 tensor_ops_5d.h 里实现过类似版本
    // 这里假设你有：apply_rotary_pos_emb_bshd(q,k,cos,sin)
    // 若你还没写，就沿你 Python 公式写一个 4D 版本即可。
    apply_rotary_pos_emb_bshd(ws->q, B, S, H, D,
                              ws->k, B, S, KVH, D,
                              rope_cos + past * D, rope_sin + past * D, S, D);

    // 5) KV cache: append current step k/v into cache
    // cache storage: (B, max_t, KVH, D)
    const int64_t T = past + S;
    if (cache != nullptr) {
        assert(cache->max_t >= T);
        kv_cache_append_btkd(cache->k[layer_id], B, cache->max_t, KVH, D,
                             ws->k, B, S, KVH, D,
                             past);
        kv_cache_append_btkd(cache->v[layer_id], B, cache->max_t, KVH, D,
                             ws->v, B, S, KVH, D,
                             past);
    }

    // 6) Build k_cat/v_cat views (B,T,KVH,D)
    const float *k_cat = (cache != nullptr) ? cache->k[layer_id] : ws->k; // if no cache, treat ws->k as (B,S,KVH,D)
    const float *v_cat = (cache != nullptr) ? cache->v[layer_id] : ws->v;
    const int64_t cat_T = (cache != nullptr) ? T : S;

    // 7) repeat kv -> (B,T,H,D)  then transpose to (B,H,T,D)
    repeat_kv_btkd_to_bthd(k_cat, B, cat_T, KVH, D, cfg.n_heads / cfg.n_kv_heads,
                           ws->k_rep, B, cat_T, H, D);
    repeat_kv_btkd_to_bthd(v_cat, B, cat_T, KVH, D, cfg.n_heads / cfg.n_kv_heads,
                           ws->v_rep, B, cat_T, H, D);

    // transpose q: (B,S,H,D)->(B,H,S,D)
    transpose_bshd_to_bhsd(ws->q, B, S, H, D,
                           ws->q_bhsd, B, H, S, D);

    // transpose k/v: (B,T,H,D)->(B,H,T,D)
    transpose_bthd_to_bhtd(ws->k_rep, B, cat_T, H, D,
                           ws->k_bhtd, B, H, cat_T, D);
    transpose_bthd_to_bhtd(ws->v_rep, B, cat_T, H, D,
                           ws->v_bhtd, B, H, cat_T, D);

    // 8) attention: scores = q @ k^T
    attn_qk_matmul(ws->q_bhsd, B, H, S, D,
                   ws->k_bhtd, B, H, cat_T, D,
                   1.0f / std::sqrt(static_cast<float>(D)),
                   ws->scores, B, H, S, cat_T);

    // 9) causal mask with prefix_offset=past
    apply_causal_mask(ws->scores, B, H, S, cat_T, past);

    // 10) padding mask (B,T) if provided
    if (attn_mask_bt != nullptr) {
        assert(m1 == B && m0 == cat_T);
        add_attention_mask(ws->scores, B, H, S, cat_T, attn_mask_bt, m1, m0);
    }

    // 11) softmax over last dim T
    attn_softmax_scores(ws->scores, B, H, S, cat_T, ws->probs, B, H, S, cat_T);

    // 12) attn_out = probs @ v
    attn_pv_matmul(ws->probs, B, H, S, cat_T,
                   ws->v_bhtd, B, H, cat_T, D,
                   ws->attn_out, B, H, S, D);

    // 13) back to (B,S,H,D) then merge heads -> (B,S,H*D)
    transpose_bhsd_to_bshd(ws->attn_out, B, H, S, D,
                           ws->attn_out_bshd, B, S, H, D);

    // merge heads copy (B,S,H,D)->(B,S,H*D)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            const float *src = ws->attn_out_bshd + off_4d(b, s, 0, 0, S, H, D);
            float *dst = ws->attn_out_flat + (b * S + s) * (H * D);
            std::memcpy(dst, src, static_cast<size_t>(H * D) * sizeof(float));
        }
    }

    // 14) o_proj: (B,S,H*D)->(B,S,hidden)
    matmul_nt(ws->attn_out_flat, B * S, H * D, lw.w_o, HIDDEN, H * D, ws->h1, B * S, HIDDEN);
    if (lw.b_o) add_bias_lastdim_inplace(ws->h1, B * S, HIDDEN, lw.b_o, HIDDEN);

    // 15) residual: out = x + o_proj
    std::memcpy(out, x, static_cast<size_t>(B * S * HIDDEN) * sizeof(float));
    add_inplace(out, B * S * HIDDEN, ws->h1, B * S * HIDDEN, 1.0f);
}

inline void minimind_ffn_forward_infer(
    const minimind_config &cfg,
    const minimind_layer_weights &lw,
    const float *x, int64_t B, int64_t S, int64_t HIDDEN, // x (B,S,hidden)
    minimind_workspace *ws,
    float *out) {
    // out (B,S,hidden)
    // pre-ffn rmsnorm
    rms_norm_lastdim(x, B * S, HIDDEN, lw.rms_ffn, HIDDEN, cfg.rms_eps, ws->h1);

    // gate/up: (B,S,hidden)->(B,S,inter)
    matmul_nt(ws->h1, B * S, HIDDEN, lw.w_gate, cfg.inter, HIDDEN, ws->ffn_gate, B * S, cfg.inter);
    if (lw.b_gate) add_bias_lastdim_inplace(ws->ffn_gate, B * S, cfg.inter, lw.b_gate, cfg.inter);

    matmul_nt(ws->h1, B * S, HIDDEN, lw.w_up, cfg.inter, HIDDEN, ws->ffn_up, B * S, cfg.inter);
    if (lw.b_up) add_bias_lastdim_inplace(ws->ffn_up, B * S, cfg.inter, lw.b_up, cfg.inter);

    // silu(gate) * up  -> mid
    silu_inplace(ws->ffn_gate, B * S * cfg.inter);
    mul(ws->ffn_gate, B * S * cfg.inter, ws->ffn_up, B * S * cfg.inter, ws->ffn_mid, B * S * cfg.inter);

    // down: (B,S,inter)->(B,S,hidden)
    matmul_nt(ws->ffn_mid, B * S, cfg.inter, lw.w_down, HIDDEN, cfg.inter, ws->ffn_out, B * S, HIDDEN);
    if (lw.b_down) add_bias_lastdim_inplace(ws->ffn_out, B * S, HIDDEN, lw.b_down, HIDDEN);

    // residual
    std::memcpy(out, x, static_cast<size_t>(B * S * HIDDEN) * sizeof(float));
    add_inplace(out, B * S * HIDDEN, ws->ffn_out, B * S * HIDDEN, 1.0f);
}

// Full forward (inference):
// input_ids (B,S) -> logits (B,S,V)  + optional KV cache update
inline void minimind_forward_infer(
    const minimind_config &cfg,
    const minimind_weights &w,
    const int32_t *input_ids, int64_t B, int64_t S,
    const uint8_t *attention_mask_bt, int64_t m1, int64_t m0, // optional (B,T)
    kv_cache *cache, // optional
    minimind_workspace *ws,
    float *out_logits, int64_t o2, int64_t o1, int64_t o0) {
    // (B,S,V)
    assert(o2 == B && o1 == S && o0 == cfg.vocab_size);
    // assert(cfg.hidden == w.layers[0].rms_attn ? cfg.hidden : cfg.hidden); // dummy "touch" to avoid unused warning

    // 0) embed -> ws->h0 (B,S,hidden)
    embedding_lookup_bsh(w.tok_embedding, cfg.vocab_size, cfg.hidden,
                         input_ids, B, S,
                         ws->h0, B, S, cfg.hidden);

    // 1) layer stack
    for (int64_t l = 0; l < cfg.n_layers; ++l) {
        // attention
        minimind_attention_forward_infer(cfg, w.layers[l],
                                         w.rope_cos, w.rope_sin, cfg.max_pos,
                                         attention_mask_bt, m1, m0,
                                         cache, static_cast<int>(l),
                                         ws->h0, B, S, cfg.hidden,
                                         ws, ws->h1);

        // ffn (dense; moe 你可以在这里替换成 moe 版本)
        minimind_ffn_forward_infer(cfg, w.layers[l],
                                   ws->h1, B, S, cfg.hidden,
                                   ws, ws->h0);
    }

    // 2) final norm: h1 = rms_norm(h0)
    rms_norm_lastdim(ws->h0, B * S, cfg.hidden, w.final_rms, cfg.hidden, cfg.rms_eps, ws->h1);

    // 3) lm_head: logits = h1 @ W^T  where W is (V, hidden)
    // flatten to 2D: (B*S, hidden) x (V, hidden)^T => (B*S, V)
    matmul_nt(ws->h1, B * S, cfg.hidden,
              w.lm_head, cfg.vocab_size, cfg.hidden,
              out_logits, B * S, cfg.vocab_size);

    // update cache length
    if (cache != nullptr) cache->cur_t += S;
}

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "tensor_ops_5d.h"
// #include "minimind_forward_5d.h"

static bool read_file(const std::string &path, std::vector<uint8_t> *out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.seekg(0, std::ios::end);
    const std::streamsize n = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out->assign(static_cast<size_t>(n), 0);
    if (!ifs.read(reinterpret_cast<char *>(out->data()), n)) return false;
    return true;
}

static bool read_f32(const std::string &path, std::vector<float> *out, size_t expect_elems) {
    std::vector<uint8_t> buf;
    if (!read_file(path, &buf)) return false;
    if (buf.size() != expect_elems * sizeof(float)) {
        std::cerr << "Size mismatch: " << path << " bytes=" << buf.size()
                << " expect=" << (expect_elems * sizeof(float)) << "\n";
        return false;
    }
    out->resize(expect_elems);
    std::memcpy(out->data(), buf.data(), buf.size());
    return true;
}

static bool read_i32(const std::string &path, std::vector<int32_t> *out, size_t expect_elems) {
    std::vector<uint8_t> buf;
    if (!read_file(path, &buf)) return false;
    if (buf.size() != expect_elems * sizeof(int32_t)) return false;
    out->resize(expect_elems);
    std::memcpy(out->data(), buf.data(), buf.size());
    return true;
}

static bool read_u8(const std::string &path, std::vector<uint8_t> *out, size_t expect_elems) {
    std::vector<uint8_t> buf;
    if (!read_file(path, &buf)) return false;
    if (buf.size() != expect_elems * sizeof(uint8_t)) return false;
    *out = std::move(buf);
    return true;
}

static void write_f32(const std::string &path, const float *x, size_t n) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(x), static_cast<std::streamsize>(n * sizeof(float)));
}

int main(int argc, char **argv) {
    const std::string dump_dir = (argc >= 2) ? argv[1] : "/Users/hastws/work_space_github_myself/CelerInfer/dump_minimind";

    // ===== 读取 meta.txt 手动对齐（这里直接 hardcode，和 Python 默认一致）=====
    // 你如果改了 Python 的环境变量，就把这里跟着改。
    const int64_t B = 2;
    const int64_t S = 5;
    const int64_t vocab = 128;
    const int64_t hidden = 64;
    const int64_t layers = 2;
    const int64_t heads = 8;
    const int64_t kv_heads = 2;
    const int64_t head_dim = hidden / heads;
    const int64_t inter = 192; // Python 自动算出来的 intermediate_size（注意：换 hidden 会变）
    const int64_t max_pos = 128;

    // ===== 读输入 =====
    std::vector<int32_t> input_ids;
    std::vector<uint8_t> attn_mask;
    if (!read_i32(dump_dir + "/input_ids_i32.bin", &input_ids, static_cast<size_t>(B * S))) {
        std::cerr << "Failed to read input_ids\n";
        return 1;
    }
    if (!read_u8(dump_dir + "/attn_mask_u8.bin", &attn_mask, static_cast<size_t>(B * S))) {
        std::cerr << "Failed to read attn_mask\n";
        return 1;
    }

    // ===== 读 rope cos/sin =====
    std::vector<float> rope_cos, rope_sin;
    if (!read_f32(dump_dir + "/rope_cos_f32.bin", &rope_cos, static_cast<size_t>(max_pos * head_dim))) {
        std::cerr << "Failed to read rope_cos\n";
        return 1;
    }
    if (!read_f32(dump_dir + "/rope_sin_f32.bin", &rope_sin, static_cast<size_t>(max_pos * head_dim))) {
        std::cerr << "Failed to read rope_sin\n";
        return 1;
    }

    // ===== 读 embedding / final norm / lm_head =====
    std::vector<float> tok_embedding, final_rms, lm_head;
    if (!read_f32(dump_dir + "/tok_embedding_f32.bin", &tok_embedding, static_cast<size_t>(vocab * hidden))) {
        std::cerr << "Failed tok_embedding\n";
        return 1;
    }
    if (!read_f32(dump_dir + "/final_rms_f32.bin", &final_rms, static_cast<size_t>(hidden))) {
        std::cerr << "Failed final_rms\n";
        return 1;
    }
    if (!read_f32(dump_dir + "/lm_head_f32.bin", &lm_head, static_cast<size_t>(vocab * hidden))) {
        std::cerr << "Failed lm_head\n";
        return 1;
    }

    // ===== 读每层权重 =====
    std::vector<minimind_layer_weights> layer_w(static_cast<size_t>(layers));
    // 每个 weight 实际数据放这里（vector 不能被 move 后地址变）
    std::vector<std::vector<float> > wq(layers), wk(layers), wv(layers), wo(layers);
    std::vector<std::vector<float> > wgate(layers), wup(layers), wdown(layers);
    std::vector<std::vector<float> > rms_attn(layers), rms_ffn(layers);

    for (int64_t l = 0; l < layers; ++l) {
        auto L = std::to_string(l);

        if (!read_f32(dump_dir + "/layer" + L + "_rms_attn.bin", &rms_attn[l], hidden)) return 1;
        if (!read_f32(dump_dir + "/layer" + L + "_rms_ffn.bin", &rms_ffn[l], hidden)) return 1;

        if (!read_f32(dump_dir + "/layer" + L + "_wq.bin", &wq[l],
                      static_cast<size_t>(heads * head_dim * hidden))) return 1;
        if (!read_f32(dump_dir + "/layer" + L + "_wk.bin", &wk[l],
                      static_cast<size_t>(kv_heads * head_dim * hidden))) return 1;
        if (!read_f32(dump_dir + "/layer" + L + "_wv.bin", &wv[l],
                      static_cast<size_t>(kv_heads * head_dim * hidden))) return 1;
        if (!read_f32(dump_dir + "/layer" + L + "_wo.bin", &wo[l],
                      static_cast<size_t>(hidden * (heads * head_dim)))) return 1;

        if (!read_f32(dump_dir + "/layer" + L + "_w_gate.bin", &wgate[l], static_cast<size_t>(inter * hidden))) return
                1;
        if (!read_f32(dump_dir + "/layer" + L + "_w_up.bin", &wup[l], static_cast<size_t>(inter * hidden))) return 1;
        if (!read_f32(dump_dir + "/layer" + L + "_w_down.bin", &wdown[l], static_cast<size_t>(hidden * inter))) return
                1;

        layer_w[l].rms_attn = rms_attn[l].data();
        layer_w[l].rms_ffn = rms_ffn[l].data();

        layer_w[l].w_q = wq[l].data();
        layer_w[l].w_k = wk[l].data();
        layer_w[l].w_v = wv[l].data();
        layer_w[l].w_o = wo[l].data();

        layer_w[l].w_gate = wgate[l].data();
        layer_w[l].w_up = wup[l].data();
        layer_w[l].w_down = wdown[l].data();

        // 本次测试 bias 都是 nullptr（Python 里 bias=False）
    }

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
    const int64_t T = S; // 本次不测 cache，T=S
    minimind_workspace ws;

    std::vector<float> h0(B * S * hidden), h1(B * S * hidden);
    std::vector<float> q(B * S * heads * head_dim);
    std::vector<float> k(B * S * kv_heads * head_dim);
    std::vector<float> v(B * S * kv_heads * head_dim);

    std::vector<float> k_rep(B * T * heads * head_dim);
    std::vector<float> v_rep(B * T * heads * head_dim);

    std::vector<float> q_bhsd(B * heads * S * head_dim);
    std::vector<float> k_bhtd(B * heads * T * head_dim);
    std::vector<float> v_bhtd(B * heads * T * head_dim);

    std::vector<float> scores(B * heads * S * T);
    std::vector<float> probs(B * heads * S * T);
    std::vector<float> attn_out(B * heads * S * head_dim);

    std::vector<float> attn_out_bshd(B * S * heads * head_dim);
    std::vector<float> attn_out_flat(B * S * heads * head_dim);

    std::vector<float> ffn_gate(B * S * inter);
    std::vector<float> ffn_up(B * S * inter);
    std::vector<float> ffn_mid(B * S * inter);
    std::vector<float> ffn_out(B * S * hidden);

    std::vector<float> logits(B * S * vocab);

    ws.h0 = h0.data();
    ws.h1 = h1.data();
    ws.q = q.data();
    ws.k = k.data();
    ws.v = v.data();
    ws.k_rep = k_rep.data();
    ws.v_rep = v_rep.data();
    ws.q_bhsd = q_bhsd.data();
    ws.k_bhtd = k_bhtd.data();
    ws.v_bhtd = v_bhtd.data();
    ws.scores = scores.data();
    ws.probs = probs.data();
    ws.attn_out = attn_out.data();
    ws.attn_out_bshd = attn_out_bshd.data();
    ws.attn_out_flat = attn_out_flat.data();
    ws.ffn_gate = ffn_gate.data();
    ws.ffn_up = ffn_up.data();
    ws.ffn_mid = ffn_mid.data();
    ws.ffn_out = ffn_out.data();
    ws.logits = logits.data();

    // ===== 运行 forward =====
    // attention_mask_bt: (B,T) 这里 T=S
    kv_cache *cache = nullptr; // 本次先不测 cache

    minimind_forward_infer(
        cfg, W,
        input_ids.data(), B, S,
        attn_mask.data(), B, T,
        cache,
        &ws,
        logits.data(), B, S, vocab);

    // ===== 写 cpp logits =====
    write_f32(dump_dir + "/cpp_logits_f32.bin", logits.data(), static_cast<size_t>(B * S * vocab));
    std::cout << "[OK] wrote " << (dump_dir + "/cpp_logits_f32.bin") << "\n";

    // ===== 读 python logits 并算 diff =====
    std::vector<float> py_logits;
    if (read_f32(dump_dir + "/py_logits_f32.bin", &py_logits, static_cast<size_t>(B * S * vocab))) {
        double max_abs = 0.0;
        double mean_abs = 0.0;
        for (size_t i = 0; i < py_logits.size(); ++i) {
            const double d = std::abs(static_cast<double>(py_logits[i]) - static_cast<double>(logits[i]));
            max_abs = std::max(max_abs, d);
            mean_abs += d;
        }
        mean_abs /= static_cast<double>(py_logits.size());
        std::cout << "diff: max_abs=" << max_abs << " mean_abs=" << mean_abs << "\n";
    } else {
        std::cout << "py_logits_f32.bin not found, skip diff.\n";
    }

    return 0;
}

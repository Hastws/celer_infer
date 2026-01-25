import os
import struct
import numpy as np
import torch

from llm_minimind_model import MiniMindConfig, MiniMindForCausalLM


def save_f32(path: str, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32)
    x.tofile(path)


def save_i32(path: str, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.int32)
    x.tofile(path)


def save_u8(path: str, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.uint8)
    x.tofile(path)


def fill_param(param: torch.Tensor, rng: np.random.RandomState, scale: float = 0.02) -> None:
    arr = rng.standard_normal(size=tuple(param.shape)).astype(np.float32) * scale
    with torch.no_grad():
        param.copy_(torch.from_numpy(arr))


def main():
    dump_dir = os.environ.get("DUMP_DIR", "dump_minimind")
    os.makedirs(dump_dir, exist_ok=True)

    # ========= 配一套“易调试”的小配置（你要换成 512/8 层也行） =========
    seed = int(os.environ.get("SEED", "123"))
    B = int(os.environ.get("B", "2"))
    S = int(os.environ.get("S", "5"))

    cfg = MiniMindConfig(
        dropout=0.0,
        hidden_size=int(os.environ.get("HIDDEN", "64")),
        num_hidden_layers=int(os.environ.get("LAYERS", "2")),
        num_attention_heads=int(os.environ.get("HEADS", "8")),
        num_key_value_heads=int(os.environ.get("KVH", "2")),
        vocab_size=int(os.environ.get("VOCAB", "128")),
        max_position_embeddings=int(os.environ.get("MAX_POS", "128")),
        rms_norm_eps=1e-5,
        rope_theta=1_000_000.0,
        inference_rope_scaling=False,
        flash_attn=False,  # 关键：强制走 slow attention
        use_moe=False,
    )

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)  # 想更确定就单线程
    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed)

    model = MiniMindForCausalLM(cfg).eval().cpu()

    # FeedForward 会把 config.intermediate_size 补齐，这里拿到最终 inter
    inter = int(model.config.intermediate_size)

    # ========= 生成 input_ids / attention_mask =========
    input_ids = np_rng.randint(0, cfg.vocab_size, size=(B, S), dtype=np.int32)
    attn_mask = np.ones((B, S), dtype=np.uint8)  # 本次测试不带 padding，全 1

    save_i32(os.path.join(dump_dir, "input_ids_i32.bin"), input_ids)
    save_u8(os.path.join(dump_dir, "attn_mask_u8.bin"), attn_mask)

    # ========= 用同一 RNG 填充权重（保证 embedding / lm_head 完全一致） =========
    # embedding + lm_head（共享）
    fill_param(model.model.embed_tokens.weight, np_rng)
    with torch.no_grad():
        model.lm_head.weight.copy_(model.model.embed_tokens.weight)

    # layer weights
    for l in range(cfg.num_hidden_layers):
        blk = model.model.layers[l]
        fill_param(blk.input_layernorm.weight, np_rng)  # rms_attn
        fill_param(blk.post_attention_layernorm.weight, np_rng)  # rms_ffn

        attn = blk.self_attn
        fill_param(attn.q_proj.weight, np_rng)
        fill_param(attn.k_proj.weight, np_rng)
        fill_param(attn.v_proj.weight, np_rng)
        fill_param(attn.o_proj.weight, np_rng)

        mlp = blk.mlp
        # 非 MoE
        fill_param(mlp.gate_proj.weight, np_rng)
        fill_param(mlp.up_proj.weight, np_rng)
        fill_param(mlp.down_proj.weight, np_rng)

    # final norm
    fill_param(model.model.norm.weight, np_rng)

    # ========= dump rope cos/sin（来自模型缓冲区） =========
    freqs_cos = model.model.freqs_cos.detach().cpu().to(torch.float32).numpy()
    freqs_sin = model.model.freqs_sin.detach().cpu().to(torch.float32).numpy()
    # freqs_cos/sin shape: (max_pos, head_dim) 其中 head_dim = hidden/heads
    save_f32(os.path.join(dump_dir, "rope_cos_f32.bin"), freqs_cos)
    save_f32(os.path.join(dump_dir, "rope_sin_f32.bin"), freqs_sin)

    # ========= dump 所有权重（raw float32） =========
    save_f32(os.path.join(dump_dir, "tok_embedding_f32.bin"),
             model.model.embed_tokens.weight.detach().cpu().numpy())
    save_f32(os.path.join(dump_dir, "final_rms_f32.bin"),
             model.model.norm.weight.detach().cpu().numpy())
    save_f32(os.path.join(dump_dir, "lm_head_f32.bin"),
             model.lm_head.weight.detach().cpu().numpy())

    for l in range(cfg.num_hidden_layers):
        blk = model.model.layers[l]
        attn = blk.self_attn
        mlp = blk.mlp

        save_f32(os.path.join(dump_dir, f"layer{l}_rms_attn.bin"),
                 blk.input_layernorm.weight.detach().cpu().numpy())
        save_f32(os.path.join(dump_dir, f"layer{l}_rms_ffn.bin"),
                 blk.post_attention_layernorm.weight.detach().cpu().numpy())

        save_f32(os.path.join(dump_dir, f"layer{l}_wq.bin"),
                 attn.q_proj.weight.detach().cpu().numpy())
        save_f32(os.path.join(dump_dir, f"layer{l}_wk.bin"),
                 attn.k_proj.weight.detach().cpu().numpy())
        save_f32(os.path.join(dump_dir, f"layer{l}_wv.bin"),
                 attn.v_proj.weight.detach().cpu().numpy())
        save_f32(os.path.join(dump_dir, f"layer{l}_wo.bin"),
                 attn.o_proj.weight.detach().cpu().numpy())

        save_f32(os.path.join(dump_dir, f"layer{l}_w_gate.bin"),
                 mlp.gate_proj.weight.detach().cpu().numpy())
        save_f32(os.path.join(dump_dir, f"layer{l}_w_up.bin"),
                 mlp.up_proj.weight.detach().cpu().numpy())
        save_f32(os.path.join(dump_dir, f"layer{l}_w_down.bin"),
                 mlp.down_proj.weight.detach().cpu().numpy())

    # ========= 写一份纯文本 meta，给 C++ 直接 hardcode/核对 =========
    meta_txt = f"""seed={seed}
B={B}
S={S}
vocab={cfg.vocab_size}
hidden={cfg.hidden_size}
layers={cfg.num_hidden_layers}
heads={cfg.num_attention_heads}
kv_heads={cfg.num_key_value_heads}
head_dim={cfg.hidden_size // cfg.num_attention_heads}
inter={inter}
max_pos={cfg.max_position_embeddings}
"""
    with open(os.path.join(dump_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(meta_txt)

    # ========= 跑 forward =========
    input_ids_t = torch.from_numpy(input_ids).to(torch.long)
    attn_mask_t = torch.from_numpy(attn_mask).to(torch.long)  # python 里 attention_mask 用 0/1 float/long 都行

    with torch.no_grad():
        out = model(input_ids=input_ids_t, attention_mask=attn_mask_t,
                    use_cache=False, logits_to_keep=0)
        logits = out.logits.detach().cpu().to(torch.float32).numpy()

    save_f32(os.path.join(dump_dir, "py_logits_f32.bin"), logits)

    print("[OK] Dumped to:", dump_dir)
    print("logits shape:", logits.shape)
    print(open(os.path.join(dump_dir, "meta.txt"), "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()

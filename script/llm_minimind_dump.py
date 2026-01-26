import os
import json
import base64
import time
from typing import Any, Dict

import numpy as np
import torch

from llm_minimind_model import MiniMindConfig, MiniMindForCausalLM


def _b64_encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _encode_array(arr: np.ndarray, dtype: np.dtype, name: str = "") -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=dtype, order="C")
    raw = arr.tobytes(order="C")
    flat = arr.reshape(-1)
    preview_n = int(os.environ.get("JSON_PREVIEW_N", "16"))
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(np.dtype(dtype)),
        "encoding": "base64",
        "data": _b64_encode_bytes(raw),
        "preview": flat[: min(preview_n, flat.size)].tolist(),
    }


def _cfg_to_json_dict(cfg: MiniMindConfig, inter: int) -> Dict[str, Any]:
    return {
        "dropout": float(cfg.dropout),
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "vocab_size": int(cfg.vocab_size),
        "max_position_embeddings": int(cfg.max_position_embeddings),
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "rope_theta": float(cfg.rope_theta),
        "inference_rope_scaling": bool(cfg.inference_rope_scaling),
        "flash_attn": bool(cfg.flash_attn),
        "use_moe": bool(cfg.use_moe),
        "intermediate_size": int(inter),
    }


def _fill_param(
    param: torch.Tensor, rng: np.random.RandomState, scale: float = 0.02
) -> None:
    arr = rng.standard_normal(size=tuple(param.shape)).astype(np.float32) * scale
    with torch.no_grad():
        param.copy_(torch.from_numpy(arr))


def main() -> None:
    out_dir = os.environ.get("DUMP_DIR", "dump_minimind")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.environ.get("JSON_PATH", os.path.join(out_dir, "minimind.json"))

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
        flash_attn=False,
        use_moe=False,
    )

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed)

    model = MiniMindForCausalLM(cfg).eval().cpu()

    inter = int(model.config.intermediate_size)
    head_dim = int(cfg.hidden_size // cfg.num_attention_heads)

    # inputs
    input_ids = np_rng.randint(0, cfg.vocab_size, size=(B, S), dtype=np.int32)
    attn_mask = np.ones((B, S), dtype=np.uint8)

    # weights (deterministic)
    _fill_param(model.model.embed_tokens.weight, np_rng)
    with torch.no_grad():
        model.lm_head.weight.copy_(model.model.embed_tokens.weight)

    for l in range(cfg.num_hidden_layers):
        blk = model.model.layers[l]
        _fill_param(blk.input_layernorm.weight, np_rng)
        _fill_param(blk.post_attention_layernorm.weight, np_rng)

        attn = blk.self_attn
        _fill_param(attn.q_proj.weight, np_rng)
        _fill_param(attn.k_proj.weight, np_rng)
        _fill_param(attn.v_proj.weight, np_rng)
        _fill_param(attn.o_proj.weight, np_rng)

        mlp = blk.mlp
        _fill_param(mlp.gate_proj.weight, np_rng)
        _fill_param(mlp.up_proj.weight, np_rng)
        _fill_param(mlp.down_proj.weight, np_rng)

    _fill_param(model.model.norm.weight, np_rng)

    # rope buffers
    freqs_cos = (
        model.model.freqs_cos.detach().cpu().to(torch.float32).numpy()
    )  # (max_pos, head_dim)
    freqs_sin = model.model.freqs_sin.detach().cpu().to(torch.float32).numpy()

    # reference logits (not timed)
    input_ids_t = torch.from_numpy(input_ids).to(torch.long)
    attn_mask_t = torch.from_numpy(attn_mask).to(torch.long)
    with torch.no_grad():
        out = model(
            input_ids=input_ids_t,
            attention_mask=attn_mask_t,
            use_cache=False,
            logits_to_keep=0,
        )
        logits = out.logits.detach().cpu().to(torch.float32).numpy()  # (B,S,V)

    # build json
    j: Dict[str, Any] = {
        "meta": {
            "seed": seed,
            "B": B,
            "S": S,
            "head_dim": head_dim,
        },
        "config": _cfg_to_json_dict(cfg, inter),
        "inputs": {
            "input_ids": _encode_array(input_ids, np.int32, "input_ids"),
            "attention_mask": _encode_array(attn_mask, np.uint8, "attention_mask"),
        },
        "rope": {
            "cos": _encode_array(freqs_cos, np.float32, "rope_cos"),
            "sin": _encode_array(freqs_sin, np.float32, "rope_sin"),
        },
        "weights": {
            "tok_embedding": _encode_array(
                model.model.embed_tokens.weight.detach().cpu().numpy(),
                np.float32,
                "tok_embedding",
            ),
            "final_rms": _encode_array(
                model.model.norm.weight.detach().cpu().numpy(), np.float32, "final_rms"
            ),
            "lm_head": _encode_array(
                model.lm_head.weight.detach().cpu().numpy(), np.float32, "lm_head"
            ),
            "layers": [],
        },
        "reference": {
            "logits": _encode_array(logits, np.float32, "reference_logits"),
        },
    }

    for l in range(cfg.num_hidden_layers):
        blk = model.model.layers[l]
        attn = blk.self_attn
        mlp = blk.mlp
        j["weights"]["layers"].append(
            {
                "rms_attn": _encode_array(
                    blk.input_layernorm.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_rms_attn",
                ),
                "rms_ffn": _encode_array(
                    blk.post_attention_layernorm.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_rms_ffn",
                ),
                "wq": _encode_array(
                    attn.q_proj.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_wq",
                ),
                "wk": _encode_array(
                    attn.k_proj.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_wk",
                ),
                "wv": _encode_array(
                    attn.v_proj.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_wv",
                ),
                "wo": _encode_array(
                    attn.o_proj.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_wo",
                ),
                "w_gate": _encode_array(
                    mlp.gate_proj.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_w_gate",
                ),
                "w_up": _encode_array(
                    mlp.up_proj.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_w_up",
                ),
                "w_down": _encode_array(
                    mlp.down_proj.weight.detach().cpu().numpy(),
                    np.float32,
                    f"layer{l}_w_down",
                ),
            }
        )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", out_json)
    print("reference logits shape:", logits.shape)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Load a JSON model (from generate_random_model.py) and run forward pass.
Measures timing (excluding JSON loading).
Saves logits for comparison/verification.
"""

import os
import json
import base64
import time
import sys
from typing import Any, Dict

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from python.core.minimind_model import MiniMindConfig, MiniMindForCausalLM


def _b64_decode_to_numpy(t: Dict[str, Any]) -> np.ndarray:
    assert t["encoding"] == "base64"
    raw = base64.b64decode(t["data"])
    dtype = np.dtype(t["dtype"])
    shape = tuple(int(x) for x in t["shape"])
    arr = np.frombuffer(raw, dtype=dtype).copy()
    return arr.reshape(shape)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_cfg(cfg_j: Dict[str, Any]) -> MiniMindConfig:
    cfg = MiniMindConfig(
        dropout=float(cfg_j["dropout"]),
        hidden_size=int(cfg_j["hidden_size"]),
        num_hidden_layers=int(cfg_j["num_hidden_layers"]),
        num_attention_heads=int(cfg_j["num_attention_heads"]),
        num_key_value_heads=int(cfg_j["num_key_value_heads"]),
        vocab_size=int(cfg_j["vocab_size"]),
        max_position_embeddings=int(cfg_j["max_position_embeddings"]),
        rms_norm_eps=float(cfg_j["rms_norm_eps"]),
        rope_theta=float(cfg_j["rope_theta"]),
        inference_rope_scaling=bool(cfg_j["inference_rope_scaling"]),
        flash_attn=bool(cfg_j["flash_attn"]),
        use_moe=bool(cfg_j["use_moe"]),
        intermediate_size=int(cfg_j.get("intermediate_size", 0)),
    )
    return cfg


def main() -> None:
    json_path = os.environ.get("JSON_PATH", "dump_minimind/minimind.json")
    dump_dir = os.environ.get("DUMP_DIR", os.path.dirname(json_path) or ".")
    os.makedirs(dump_dir, exist_ok=True)

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    # --------- load json (NOT timed) ----------
    print(f"Loading JSON from: {json_path}")
    j = _load_json(json_path)

    cfg = _build_cfg(j["config"])
    model = MiniMindForCausalLM(cfg).eval().cpu()

    print(
        f"Config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
        f"heads={cfg.num_attention_heads}, vocab={cfg.vocab_size}"
    )

    # inputs
    input_ids = _b64_decode_to_numpy(j["inputs"]["input_ids"]).astype(np.int32)
    attn_mask = _b64_decode_to_numpy(j["inputs"]["attention_mask"]).astype(np.uint8)

    input_ids_t = torch.from_numpy(input_ids).to(torch.long)
    attn_mask_t = torch.from_numpy(attn_mask.astype(np.int64))

    # weights (NOT timed)
    w = j["weights"]
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(
            torch.from_numpy(
                _b64_decode_to_numpy(w["tok_embedding"]).astype(np.float32)
            )
        )
        model.model.norm.weight.copy_(
            torch.from_numpy(_b64_decode_to_numpy(w["final_rms"]).astype(np.float32))
        )
        model.lm_head.weight.copy_(
            torch.from_numpy(_b64_decode_to_numpy(w["lm_head"]).astype(np.float32))
        )

        for l, layer_j in enumerate(w["layers"]):
            blk = model.model.layers[l]
            attn = blk.self_attn
            mlp = blk.mlp

            blk.input_layernorm.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["rms_attn"]).astype(np.float32)
                )
            )
            blk.post_attention_layernorm.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["rms_ffn"]).astype(np.float32)
                )
            )

            attn.q_proj.weight.copy_(
                torch.from_numpy(_b64_decode_to_numpy(layer_j["wq"]).astype(np.float32))
            )
            attn.k_proj.weight.copy_(
                torch.from_numpy(_b64_decode_to_numpy(layer_j["wk"]).astype(np.float32))
            )
            attn.v_proj.weight.copy_(
                torch.from_numpy(_b64_decode_to_numpy(layer_j["wv"]).astype(np.float32))
            )
            attn.o_proj.weight.copy_(
                torch.from_numpy(_b64_decode_to_numpy(layer_j["wo"]).astype(np.float32))
            )

            mlp.gate_proj.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["w_gate"]).astype(np.float32)
                )
            )
            mlp.up_proj.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["w_up"]).astype(np.float32)
                )
            )
            mlp.down_proj.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["w_down"]).astype(np.float32)
                )
            )

        # rope (optional: overwrite buffers to guarantee exact)
        rope_cos = _b64_decode_to_numpy(j["rope"]["cos"]).astype(np.float32)
        rope_sin = _b64_decode_to_numpy(j["rope"]["sin"]).astype(np.float32)
        model.model.freqs_cos.copy_(torch.from_numpy(rope_cos))
        model.model.freqs_sin.copy_(torch.from_numpy(rope_sin))

    print("[OK] Weights loaded from JSON")

    # --------- forward timed only ----------
    warmup = int(os.environ.get("WARMUP", "1"))
    if warmup > 0:
        print(f"Running {warmup} warmup iterations...")
        with torch.no_grad():
            _ = model(
                input_ids=input_ids_t,
                attention_mask=attn_mask_t,
                use_cache=False,
                logits_to_keep=0,
            )

    # Time the forward pass (excluding JSON loading)
    print("Running timed forward pass...")
    start_time = time.time()
    with torch.no_grad():
        out = model(
            input_ids=input_ids_t,
            attention_mask=attn_mask_t,
            use_cache=False,
            logits_to_keep=0,
        )
    elapsed = time.time() - start_time

    logits = out.logits.detach().cpu().to(torch.float32).numpy()  # (B, S, V)

    # Print results
    print()
    print("=" * 60)
    print("[Forward] Shape: {}, Dtype: {}".format(logits.shape, logits.dtype))
    print("[Timing] Forward pass: {:.2f}ms (warmup={})".format(elapsed * 1000, warmup))
    print(
        "[Logits] Min: {:.6f}, Max: {:.6f}, Mean: {:.6f}".format(
            logits.min(), logits.max(), logits.mean()
        )
    )
    print("=" * 60)
    print()

    # Save logits
    logits_path = os.path.join(dump_dir, "logits_torch.npy")
    np.save(logits_path, logits)
    print("[OK] Saved logits to: {}".format(logits_path))

    # Save embedding output for debugging
    print("\n[DEBUG] Extracting embedding output for comparison...")
    with torch.no_grad():
        # Get embedding layer output (model.model.embed_tokens)
        h0_emb = model.model.embed_tokens(input_ids_t)  # (B, S, hidden)
        # Also apply dropout like the model does
        h0 = model.model.dropout(h0_emb)
        h0_np = h0.cpu().numpy()
        h0_path = os.path.join(dump_dir, "h0_torch.npy")
        np.save(h0_path, h0_np)
        print(
            "[DEBUG] Saved h0 (embedding output after dropout) to: {}".format(h0_path)
        )
        print(
            "[DEBUG] h0 shape: {}, min: {:.6f}, max: {:.6f}, mean: {:.6f}".format(
                h0_np.shape, h0_np.min(), h0_np.max(), h0_np.mean()
            )
        )

    # Save intermediate layer outputs using hooks
    print(
        "[DEBUG] Layer 0 intermediate outputs - skipping (requires model modification)"
    )


if __name__ == "__main__":
    main()

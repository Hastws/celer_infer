#!/usr/bin/env python3
"""
Simpler debug script - just run forward and extract layer 0 outputs via module hooks.
"""

import os
import json
import base64
from typing import Any, Dict

import numpy as np
import torch

from llm_minimind_model import MiniMindConfig, MiniMindForCausalLM


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


def main():
    dump_dir = os.environ.get("DUMP_DIR", "dump_minimind")
    json_path = os.environ.get("JSON_PATH", f"{dump_dir}/minimind.json")
    device = "cpu"

    print(f"Loading JSON from: {json_path}")
    j = _load_json(json_path)
    cfg = _build_cfg(j["config"])

    # Create model and load weights
    model = MiniMindForCausalLM(cfg)
    model.eval()
    model.to(device)

    # Load ALL weights from JSON
    with torch.no_grad():
        # Embedding
        tok_emb = torch.from_numpy(
            _b64_decode_to_numpy(j["weights"]["tok_embedding"]).astype(np.float32)
        )
        model.model.embed_tokens.weight.copy_(tok_emb)

        # Final norm and LM head
        final_rms = torch.from_numpy(
            _b64_decode_to_numpy(j["weights"]["final_rms"]).astype(np.float32)
        )
        model.model.norm.weight.copy_(final_rms)

        lm_head = torch.from_numpy(
            _b64_decode_to_numpy(j["weights"]["lm_head"]).astype(np.float32)
        )
        model.lm_head.weight.copy_(lm_head)

        # Rope
        rope_cos = torch.from_numpy(
            _b64_decode_to_numpy(j["rope"]["cos"]).astype(np.float32)
        )
        rope_sin = torch.from_numpy(
            _b64_decode_to_numpy(j["rope"]["sin"]).astype(np.float32)
        )
        model.model.freqs_cos.copy_(rope_cos)
        model.model.freqs_sin.copy_(rope_sin)

        # Layer weights
        for l in range(cfg.num_hidden_layers):
            layer = model.model.layers[l]
            layer_j = j["weights"]["layers"][l]

            layer.input_layernorm.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["rms_attn"]).astype(np.float32)
                )
            )
            layer.post_attention_layernorm.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["rms_ffn"]).astype(np.float32)
                )
            )

            layer.self_attn.q_proj.weight.copy_(
                torch.from_numpy(_b64_decode_to_numpy(layer_j["wq"]).astype(np.float32))
            )
            layer.self_attn.k_proj.weight.copy_(
                torch.from_numpy(_b64_decode_to_numpy(layer_j["wk"]).astype(np.float32))
            )
            layer.self_attn.v_proj.weight.copy_(
                torch.from_numpy(_b64_decode_to_numpy(layer_j["wv"]).astype(np.float32))
            )
            layer.self_attn.o_proj.weight.copy_(
                torch.from_numpy(_b64_decode_to_numpy(layer_j["wo"]).astype(np.float32))
            )

            layer.mlp.gate_proj.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["w_gate"]).astype(np.float32)
                )
            )
            layer.mlp.up_proj.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["w_up"]).astype(np.float32)
                )
            )
            layer.mlp.down_proj.weight.copy_(
                torch.from_numpy(
                    _b64_decode_to_numpy(layer_j["w_down"]).astype(np.float32)
                )
            )

    # Prepare inputs
    input_ids = _b64_decode_to_numpy(j["inputs"]["input_ids"]).astype(np.int32)
    attn_mask = _b64_decode_to_numpy(j["inputs"]["attention_mask"]).astype(np.uint8)
    input_ids_t = torch.from_numpy(input_ids.reshape(2, 5)).to(
        dtype=torch.long, device=device
    )
    attn_mask_t = torch.from_numpy(attn_mask.reshape(2, 5)).to(
        dtype=torch.uint8, device=device
    )

    print("[DEBUG] Extracting layer 0 intermediate outputs using hooks...")

    # Storage for hook outputs
    hook_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            hook_outputs[name] = (
                output.detach().clone() if isinstance(output, torch.Tensor) else output
            )

        return hook

    # Register hooks
    layer0 = model.model.layers[0]
    h_input_hook = layer0.input_layernorm.register_forward_hook(
        make_hook("input_layernorm_out")
    )
    attn_hook = layer0.self_attn.register_forward_hook(make_hook("attn_out"))
    ffn_hook = layer0.mlp.register_forward_hook(make_hook("ffn_out"))

    # Run forward
    with torch.no_grad():
        out = model(
            input_ids=input_ids_t,
            attention_mask=attn_mask_t,
            use_cache=False,
            logits_to_keep=0,
        )

    # Extract and save
    if "input_layernorm_out" in hook_outputs:
        h_norm = hook_outputs["input_layernorm_out"]
        if isinstance(h_norm, tuple):
            h_norm = h_norm[0] if len(h_norm) > 0 else h_norm
        h_norm_np = h_norm.cpu().numpy()
        path = os.path.join(dump_dir, "h_norm_l0_torch.npy")
        np.save(path, h_norm_np)
        print(f"[DEBUG] Saved h_norm_l0 to: {path}")
        print(
            f"        Stats: min={h_norm_np.min():.6f}, max={h_norm_np.max():.6f}, mean={h_norm_np.mean():.6f}"
        )

    if "attn_out" in hook_outputs:
        attn_out_tuple = hook_outputs["attn_out"]
        if isinstance(attn_out_tuple, tuple):
            attn_out = attn_out_tuple[0]
        else:
            attn_out = attn_out_tuple
        attn_out_np = attn_out.cpu().numpy()
        path = os.path.join(dump_dir, "h1_attn_l0_torch.npy")
        np.save(path, attn_out_np)
        print(f"[DEBUG] Saved h1_attn_l0 to: {path}")
        print(
            f"        Stats: min={attn_out_np.min():.6f}, max={attn_out_np.max():.6f}, mean={attn_out_np.mean():.6f}"
        )

    if "ffn_out" in hook_outputs:
        ffn_out = hook_outputs["ffn_out"]
        ffn_out_np = ffn_out.cpu().numpy()
        path = os.path.join(dump_dir, "h0_ffn_l0_torch.npy")
        np.save(path, ffn_out_np)
        print(f"[DEBUG] Saved h0_ffn_l0 to: {path}")
        print(
            f"        Stats: min={ffn_out_np.min():.6f}, max={ffn_out_np.max():.6f}, mean={ffn_out_np.mean():.6f}"
        )

    print("\n[DEBUG] Layer 0 intermediate extraction complete!")


if __name__ == "__main__":
    main()

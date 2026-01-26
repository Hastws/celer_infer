#!/usr/bin/env python3
"""
Extract layer 0 intermediate outputs from PyTorch with detailed attention debugging.
"""

import torch
import torch.nn.functional as F
import json
import base64
import numpy as np
import os
from pathlib import Path

# Import model
import sys

sys.path.insert(0, "script")
from llm_minimind_model import MiniMindForCausalLM, MiniMindConfig


def b64_decode(b64_str):
    """Decode base64 string to bytes."""
    return base64.b64decode(b64_str)


def load_tensor_from_json(data_dict):
    """Load tensor from JSON dict with base64-encoded data."""
    b64_data = data_dict["data"]
    decoded = b64_decode(b64_data)
    shape = data_dict["shape"]
    dtype_str = data_dict.get("dtype", "float32")

    if dtype_str == "float32":
        arr = np.frombuffer(decoded, dtype=np.float32)
    elif dtype_str == "int32":
        arr = np.frombuffer(decoded, dtype=np.int32)
    elif dtype_str == "uint8":
        arr = np.frombuffer(decoded, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    return arr.reshape(shape)


def main():
    json_path = "dump_minimind/minimind.json"
    with open(json_path) as f:
        data = json.load(f)

    cfg_dict = data["config"]
    config = MiniMindConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_size=cfg_dict["hidden_size"],
        num_hidden_layers=cfg_dict["num_hidden_layers"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        intermediate_size=cfg_dict["intermediate_size"],
        max_position_embeddings=cfg_dict["max_position_embeddings"],
    )

    print(f"[DEBUG] Loading model with config: {config}")
    model = MiniMindForCausalLM(config)
    model.eval()

    # Load weights from JSON
    print(f"[DEBUG] Loading weights from JSON...")

    # tok_embedding
    tok_emb = load_tensor_from_json(data["weights"]["tok_embedding"])
    model.model.embed_tokens.weight.data = torch.from_numpy(tok_emb).float()

    # final_rms
    final_rms = load_tensor_from_json(data["weights"]["final_rms"])
    model.model.norm.weight.data = torch.from_numpy(final_rms).float()

    # lm_head
    lm_head = load_tensor_from_json(data["weights"]["lm_head"])
    model.lm_head.weight.data = torch.from_numpy(lm_head).float()

    # rope
    rope_cos = load_tensor_from_json(data["rope"]["cos"])
    rope_sin = load_tensor_from_json(data["rope"]["sin"])
    model.model.freqs_cos = torch.from_numpy(rope_cos).float()
    model.model.freqs_sin = torch.from_numpy(rope_sin).float()

    # layers
    for l, layer_data in enumerate(data["weights"]["layers"]):
        layer = model.model.layers[l]

        # attention
        wq = load_tensor_from_json(layer_data["wq"])
        layer.self_attn.q_proj.weight.data = torch.from_numpy(wq).float()

        wk = load_tensor_from_json(layer_data["wk"])
        layer.self_attn.k_proj.weight.data = torch.from_numpy(wk).float()

        wv = load_tensor_from_json(layer_data["wv"])
        layer.self_attn.v_proj.weight.data = torch.from_numpy(wv).float()

        wo = load_tensor_from_json(layer_data["wo"])
        layer.self_attn.o_proj.weight.data = torch.from_numpy(wo).float()

        rms_attn = load_tensor_from_json(layer_data["rms_attn"])
        layer.input_layernorm.weight.data = torch.from_numpy(rms_attn).float()

        # ffn
        w_gate = load_tensor_from_json(layer_data["w_gate"])
        layer.mlp.gate_proj.weight.data = torch.from_numpy(w_gate).float()

        w_up = load_tensor_from_json(layer_data["w_up"])
        layer.mlp.up_proj.weight.data = torch.from_numpy(w_up).float()

        w_down = load_tensor_from_json(layer_data["w_down"])
        layer.mlp.down_proj.weight.data = torch.from_numpy(w_down).float()

        rms_ffn = load_tensor_from_json(layer_data["rms_ffn"])
        layer.post_attention_layernorm.weight.data = torch.from_numpy(rms_ffn).float()

    print(f"[DEBUG] Weights loaded successfully")

    # Load input
    meta = data["meta"]
    B, S = meta["B"], meta["S"]
    input_ids = torch.from_numpy(
        load_tensor_from_json(data["inputs"]["input_ids"])
    ).long()

    print(f"[DEBUG] Input shape: {input_ids.shape}")

    # Set up hooks to capture attention outputs
    layer0_outputs = {}

    def hook_input_layernorm(module, input, output):
        layer0_outputs["h_norm_l0"] = output.detach().clone()

    def hook_attention(module, input, output):
        # output is (hidden_states, ...)
        if isinstance(output, tuple):
            layer0_outputs["h1_attn_l0"] = output[0].detach().clone()
        else:
            layer0_outputs["h1_attn_l0"] = output.detach().clone()

    def hook_q_proj(module, input, output):
        # Capture Q projections (B, S, H*D)
        layer0_outputs["q_flat_l0"] = output.detach().clone()

    def hook_k_proj(module, input, output):
        # Capture K projections (B, S, KVH*D)
        layer0_outputs["k_flat_l0"] = output.detach().clone()

    def hook_v_proj(module, input, output):
        # Capture V projections (B, S, KVH*D)
        layer0_outputs["v_flat_l0"] = output.detach().clone()

    # Register hooks
    layer0 = model.model.layers[0]
    layer0.input_layernorm.register_forward_hook(hook_input_layernorm)
    layer0.self_attn.register_forward_hook(hook_attention)
    layer0.self_attn.q_proj.register_forward_hook(hook_q_proj)
    layer0.self_attn.k_proj.register_forward_hook(hook_k_proj)
    layer0.self_attn.v_proj.register_forward_hook(hook_v_proj)

    # Forward pass
    print(f"[DEBUG] Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    # Save outputs
    dump_dir = "dump_minimind"
    os.makedirs(dump_dir, exist_ok=True)

    # Save h_norm_l0
    if "h_norm_l0" in layer0_outputs:
        h_norm = layer0_outputs["h_norm_l0"].cpu().numpy().astype(np.float32)
        path = os.path.join(dump_dir, "h_norm_l0_torch.npy")
        np.save(path, h_norm, allow_pickle=False)
        stats = (
            f"min={h_norm.min():.6f}, max={h_norm.max():.6f}, mean={h_norm.mean():.6f}"
        )
        print(f"[DEBUG] Saved h_norm_l0 to: {path}")
        print(f"        Stats: {stats}")

    # Save h1_attn_l0
    if "h1_attn_l0" in layer0_outputs:
        h1_attn = layer0_outputs["h1_attn_l0"].cpu().numpy().astype(np.float32)
        path = os.path.join(dump_dir, "h1_attn_l0_torch.npy")
        np.save(path, h1_attn, allow_pickle=False)
        stats = f"min={h1_attn.min():.6f}, max={h1_attn.max():.6f}, mean={h1_attn.mean():.6f}"
        print(f"[DEBUG] Saved h1_attn_l0 to: {path}")
        print(f"        Stats: {stats}")

    # Save q_flat_l0
    if "q_flat_l0" in layer0_outputs:
        q_flat = layer0_outputs["q_flat_l0"].cpu().numpy().astype(np.float32)
        path = os.path.join(dump_dir, "q_flat_l0_torch.npy")
        np.save(path, q_flat, allow_pickle=False)
        stats = (
            f"min={q_flat.min():.6f}, max={q_flat.max():.6f}, mean={q_flat.mean():.6f}"
        )
        print(f"[DEBUG] Saved q_flat_l0 to: {path}")
        print(f"        Stats: {stats}")

    # Save k_flat_l0
    if "k_flat_l0" in layer0_outputs:
        k_flat = layer0_outputs["k_flat_l0"].cpu().numpy().astype(np.float32)
        path = os.path.join(dump_dir, "k_flat_l0_torch.npy")
        np.save(path, k_flat, allow_pickle=False)
        stats = (
            f"min={k_flat.min():.6f}, max={k_flat.max():.6f}, mean={k_flat.mean():.6f}"
        )
        print(f"[DEBUG] Saved k_flat_l0 to: {path}")
        print(f"        Stats: {stats}")

    # Save v_flat_l0
    if "v_flat_l0" in layer0_outputs:
        v_flat = layer0_outputs["v_flat_l0"].cpu().numpy().astype(np.float32)
        path = os.path.join(dump_dir, "v_flat_l0_torch.npy")
        np.save(path, v_flat, allow_pickle=False)
        stats = (
            f"min={v_flat.min():.6f}, max={v_flat.max():.6f}, mean={v_flat.mean():.6f}"
        )
        print(f"[DEBUG] Saved v_flat_l0 to: {path}")
        print(f"        Stats: {stats}")

    print(f"[DEBUG] Layer 0 intermediate extraction complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Detailed attention debugging: capture outputs after each step.
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

    model = MiniMindForCausalLM(config)
    model.eval()

    # Load weights
    tok_emb = load_tensor_from_json(data["weights"]["tok_embedding"])
    model.model.embed_tokens.weight.data = torch.from_numpy(tok_emb).float()

    final_rms = load_tensor_from_json(data["weights"]["final_rms"])
    model.model.norm.weight.data = torch.from_numpy(final_rms).float()

    lm_head = load_tensor_from_json(data["weights"]["lm_head"])
    model.lm_head.weight.data = torch.from_numpy(lm_head).float()

    rope_cos = load_tensor_from_json(data["rope"]["cos"])
    rope_sin = load_tensor_from_json(data["rope"]["sin"])
    model.model.freqs_cos = torch.from_numpy(rope_cos).float()
    model.model.freqs_sin = torch.from_numpy(rope_sin).float()

    for l, layer_data in enumerate(data["weights"]["layers"]):
        layer = model.model.layers[l]

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

        w_gate = load_tensor_from_json(layer_data["w_gate"])
        layer.mlp.gate_proj.weight.data = torch.from_numpy(w_gate).float()

        w_up = load_tensor_from_json(layer_data["w_up"])
        layer.mlp.up_proj.weight.data = torch.from_numpy(w_up).float()

        w_down = load_tensor_from_json(layer_data["w_down"])
        layer.mlp.down_proj.weight.data = torch.from_numpy(w_down).float()

        rms_ffn = load_tensor_from_json(layer_data["rms_ffn"])
        layer.post_attention_layernorm.weight.data = torch.from_numpy(rms_ffn).float()

    # Load input
    meta = data["meta"]
    B, S = meta["B"], meta["S"]
    input_ids = torch.from_numpy(
        load_tensor_from_json(data["inputs"]["input_ids"])
    ).long()

    # Patch attention forward to capture intermediate outputs
    original_forward = model.model.layers[0].self_attn.forward

    captured = {}

    def patched_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Q, K, V projections (already done in forward, recompute here for capture)
        query_states = model.model.layers[0].self_attn.q_proj(hidden_states)
        key_states = model.model.layers[0].self_attn.k_proj(hidden_states)
        value_states = model.model.layers[0].self_attn.v_proj(hidden_states)

        # Reshape to (B, S, heads, head_dim)
        query_states = query_states.view(
            bsz, q_len, model.model.layers[0].self_attn.num_heads, -1
        )
        key_states = key_states.view(
            bsz, q_len, model.model.layers[0].self_attn.num_key_value_heads, -1
        )
        value_states = value_states.view(
            bsz, q_len, model.model.layers[0].self_attn.num_key_value_heads, -1
        )

        # Capture before RoPE
        captured["q_before_rope"] = query_states.detach().clone()
        captured["k_before_rope"] = key_states.detach().clone()

        # Apply RoPE
        query_states = model.model.apply_rotary_emb(
            query_states, model.model.freqs_cos, model.model.freqs_sin
        )
        key_states = model.model.apply_rotary_emb(
            key_states, model.model.freqs_cos, model.model.freqs_sin
        )

        # Capture after RoPE
        captured["q_after_rope"] = query_states.detach().clone()
        captured["k_after_rope"] = key_states.detach().clone()

        # Repeat KV for multi-head attention
        key_states = torch.repeat_interleave(
            key_states,
            model.model.layers[0].self_attn.num_heads
            // model.model.layers[0].self_attn.num_key_value_heads,
            dim=2,
        )
        value_states = torch.repeat_interleave(
            value_states,
            model.model.layers[0].self_attn.num_heads
            // model.model.layers[0].self_attn.num_key_value_heads,
            dim=2,
        )

        # Transpose to (B, H, S, D)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (
            model.model.layers[0].self_attn.head_dim**0.5
        )

        # Capture attention scores before masking
        captured["scores_before_mask"] = attn_weights.detach().clone()

        # Apply mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Capture attention scores after masking
        captured["scores_after_mask"] = attn_weights.detach().clone()

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Capture attention weights after softmax
        captured["attn_weights"] = attn_weights.detach().clone()

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Apply output projection
        attn_output = model.model.layers[0].self_attn.o_proj(attn_output)

        return attn_output, None, None

    # Apply patch
    model.model.layers[0].self_attn.forward = patched_forward

    # Forward pass
    print(f"[DEBUG] Running forward pass with patched attention...")
    with torch.no_grad():
        outputs = model(input_ids)

    # Save outputs
    dump_dir = "dump_minimind"
    os.makedirs(dump_dir, exist_ok=True)

    print(f"\n[DEBUG] Captured attention intermediates:")
    for name, tensor in captured.items():
        arr = tensor.cpu().numpy().astype(np.float32)
        path = os.path.join(dump_dir, f"{name}_torch.npy")
        np.save(path, arr, allow_pickle=False)
        stats = f"shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
        print(f"  {name}: {stats}")
        print(f"    -> saved to {path}")


if __name__ == "__main__":
    main()

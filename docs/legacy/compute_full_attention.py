#!/usr/bin/env python3
"""
Complete manual PyTorch attention computation from scores to final output.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import base64
import os


def b64_decode(b64_str):
    return base64.b64decode(b64_str)


def load_tensor_from_json(data_dict):
    b64_data = data_dict["data"]
    decoded = b64_decode(b64_data)
    shape = data_dict["shape"]
    dtype_str = data_dict.get("dtype", "float32")

    if dtype_str == "float32":
        arr = np.frombuffer(decoded, dtype=np.float32)
    else:
        arr = np.frombuffer(
            decoded, dtype=np.int32 if dtype_str == "int32" else np.uint8
        )

    return arr.reshape(shape)


def main():
    dump_dir = "dump_minimind"

    with open(os.path.join(dump_dir, "minimind.json")) as f:
        config_data = json.load(f)

    B, S = config_data["meta"]["B"], config_data["meta"]["S"]
    hidden = config_data["config"]["hidden_size"]
    heads = config_data["config"]["num_attention_heads"]
    kv_heads = config_data["config"]["num_key_value_heads"]
    head_dim = hidden // heads

    print(f"Config: B={B}, S={S}, hidden={hidden}, heads={heads}, head_dim={head_dim}")

    # Load weights for layer 0
    layer_data = config_data["weights"]["layers"][0]
    wo_weight = load_tensor_from_json(layer_data["wo"])  # (hidden, H*D)

    # Load PyTorch scores (we know they match C++)
    scores = np.load(
        os.path.join(dump_dir, "scores_l0_torch_manual.npy")
    )  # (B, H, S, S)

    # Load v_flat and reshape
    v_flat = np.load(os.path.join(dump_dir, "v_flat_l0_torch.npy"))  # (B, S, KVH*D)
    v = v_flat.reshape(B, S, kv_heads, head_dim)  # (B, S, KVH, D)

    # Repeat and reshape for attention
    rep_factor = heads // kv_heads
    v_rep = np.repeat(v, rep_factor, axis=2)  # (B, S, H, D)
    v_bhd = v_rep.transpose(0, 2, 1, 3)  # (B, H, S, D)

    print(f"v_bhd shape: {v_bhd.shape}")

    # Convert to torch
    scores_t = torch.from_numpy(scores).float()  # (B, H, S, S)
    v_bhd_t = torch.from_numpy(v_bhd).float()  # (B, H, S, D)
    wo_weight_t = torch.from_numpy(wo_weight).float()  # (hidden, H*D)

    # Apply causal mask before softmax (same as C++)
    # Mask where future tokens should not attend to past tokens
    # Create a triangular mask: (S, S) where mask[i, j] = 0 if j > i (future)
    causal_mask = torch.tril(torch.ones(S, S)).bool()  # (S, S)
    # Convert to attention scores mask: -inf where mask is False, 0 where True
    mask_value = torch.tensor(float("-inf"))
    scores_t = scores_t.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), mask_value)

    # Softmax to get probabilities
    probs = F.softmax(scores_t, dim=-1)  # (B, H, S, S)

    print(
        f"Probs: min={probs.min():.6f}, max={probs.max():.6f}, mean={probs.mean():.6f}"
    )

    # Attention output: probs @ v
    attn_out_bhsd = torch.matmul(probs, v_bhd_t)  # (B, H, S, D)

    print(f"attn_out_bhsd shape: {attn_out_bhsd.shape}")
    print(
        f"attn_out_bhsd: min={attn_out_bhsd.min():.6f}, max={attn_out_bhsd.max():.6f}, mean={attn_out_bhsd.mean():.6f}"
    )

    # Reshape back to (B, S, H, D)
    attn_out_bshd = attn_out_bhsd.transpose(1, 2)  # (B, S, H, D)

    # Flatten to (B, S, H*D)
    attn_out_flat = attn_out_bshd.reshape(B, S, -1).float()  # (B, S, H*D)

    print(f"attn_out_flat shape: {attn_out_flat.shape}")
    print(
        f"attn_out_flat: min={attn_out_flat.min():.6f}, max={attn_out_flat.max():.6f}, mean={attn_out_flat.mean():.6f}"
    )

    # Output projection: (B, S, H*D) @ (H*D, hidden)^T -> (B, S, hidden)
    # Note: PyTorch stores weights as (out, in)
    attn_proj = torch.matmul(attn_out_flat, wo_weight_t.t())  # (B, S, hidden)

    print(f"attn_proj shape: {attn_proj.shape}")
    print(
        f"attn_proj: min={attn_proj.min():.6f}, max={attn_proj.max():.6f}, mean={attn_proj.mean():.6f}"
    )

    # Save intermediate for comparison
    np.save(
        os.path.join(dump_dir, "attn_out_flat_torch.npy"),
        attn_out_flat.numpy().astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        os.path.join(dump_dir, "attn_proj_torch.npy"),
        attn_proj.numpy().astype(np.float32),
        allow_pickle=False,
    )

    print(f"\n[DEBUG] Saved attn_out_flat and attn_proj")


if __name__ == "__main__":
    main()

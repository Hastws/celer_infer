#!/usr/bin/env python3
"""
Simple script to compute PyTorch attention scores manually and compare with C++.
"""

import torch
import numpy as np
import json
import base64
import os


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
    dump_dir = "dump_minimind"

    with open(os.path.join(dump_dir, "minimind.json")) as f:
        config_data = json.load(f)

    B, S = config_data["meta"]["B"], config_data["meta"]["S"]
    hidden = config_data["config"]["hidden_size"]
    heads = config_data["config"]["num_attention_heads"]
    kv_heads = config_data["config"]["num_key_value_heads"]
    head_dim = hidden // heads

    print(
        f"Config: B={B}, S={S}, hidden={hidden}, heads={heads}, kv_heads={kv_heads}, head_dim={head_dim}"
    )

    # Load PyTorch intermediate arrays
    q_flat = np.load(os.path.join(dump_dir, "q_flat_l0_torch.npy"))  # (B, S, H*D)
    k_flat = np.load(os.path.join(dump_dir, "k_flat_l0_torch.npy"))  # (B, S, KVH*D)
    v_flat = np.load(os.path.join(dump_dir, "v_flat_l0_torch.npy"))  # (B, S, KVH*D)

    # Load rope
    rope_cos_data = config_data["rope"]["cos"]
    rope_cos = load_tensor_from_json(rope_cos_data)  # (max_pos, head_dim)

    rope_sin_data = config_data["rope"]["sin"]
    rope_sin = load_tensor_from_json(rope_sin_data)  # (max_pos, head_dim)

    print(f"q_flat shape: {q_flat.shape}")
    print(f"k_flat shape: {k_flat.shape}")
    print(f"v_flat shape: {v_flat.shape}")
    print(f"rope_cos shape: {rope_cos.shape}")
    print(f"rope_sin shape: {rope_sin.shape}")

    # Convert to torch
    q_flat_t = torch.from_numpy(q_flat).float()  # (B, S, H*D)
    k_flat_t = torch.from_numpy(k_flat).float()  # (B, S, KVH*D)
    v_flat_t = torch.from_numpy(v_flat).float()  # (B, S, KVH*D)
    rope_cos_t = torch.from_numpy(rope_cos).float()
    rope_sin_t = torch.from_numpy(rope_sin).float()

    # Reshape Q/K/V to (B, S, heads, head_dim)
    q = q_flat_t.reshape(B, S, heads, head_dim)  # (B, S, H, D)
    k = k_flat_t.reshape(B, S, kv_heads, head_dim)  # (B, S, KVH, D)
    v = v_flat_t.reshape(B, S, kv_heads, head_dim)  # (B, S, KVH, D)

    print(f"\nAfter reshape:")
    print(f"q shape: {q.shape}")
    print(f"k shape: {k.shape}")
    print(f"v shape: {v.shape}")

    # Apply RoPE manually to match PyTorch implementation
    # PyTorch uses: (q * cos) + (rotate_half(q) * sin)
    # where rotate_half: (-x[..., D/2:], x[..., :D/2])

    cos = rope_cos_t[0:S, :head_dim]  # (S, D)
    sin = rope_sin_t[0:S, :head_dim]  # (S, D)

    # Expand for (B, S, H, D) or (B, S, KVH, D)
    # We need cos/sin to be (1, S, 1, D) for proper broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)

    # Rotate half function
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    # Apply RoPE
    q_rot = (q * cos) + (rotate_half(q) * sin)  # (B, S, H, D)
    k_rot = (k * cos) + (rotate_half(k) * sin)  # (B, S, KVH, D)

    # Repeat KV heads to match query heads
    rep_factor = heads // kv_heads
    k_rep = torch.repeat_interleave(k_rot, rep_factor, dim=2)  # (B, S, H, D)
    v_rep = torch.repeat_interleave(v, rep_factor, dim=2)  # (B, S, H, D)

    # Transpose to (B, H, S, D)
    q_bhsd = q_rot.transpose(1, 2)  # (B, H, S, D)
    k_bhsd = k_rep.transpose(1, 2)  # (B, H, S, D)
    v_bhsd = v_rep.transpose(1, 2)  # (B, H, S, D)

    # Compute attention scores
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) / (head_dim**0.5)

    print(f"\nScores shape: {scores.shape}")
    print(
        f"Scores stats: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}"
    )

    # Save computed scores
    scores_np = scores.numpy().astype(np.float32)
    np.save(
        os.path.join(dump_dir, "scores_l0_torch_manual.npy"),
        scores_np,
        allow_pickle=False,
    )

    # Load C++ scores
    cpp_scores = np.fromfile(
        os.path.join(dump_dir, "scores_l0_cpp.npy"), dtype=np.float32
    )
    cpp_scores = cpp_scores.reshape(B, heads, S, S)

    print(f"C++ scores shape: {cpp_scores.shape}")
    print(
        f"C++ scores stats: min={cpp_scores.min():.6f}, max={cpp_scores.max():.6f}, mean={cpp_scores.mean():.6f}"
    )

    # Compare
    print(f"\n{'='*80}")
    print(f"ATTENTION SCORES COMPARISON")
    print(f"{'='*80}")

    abs_diff = np.abs(scores_np - cpp_scores)
    rel_err = np.where(
        np.abs(scores_np) > 1e-6,
        np.abs(scores_np - cpp_scores) / (np.abs(scores_np) + 1e-8),
        np.abs(scores_np - cpp_scores),
    )

    print(
        f"Abs diff: max={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}, median={np.median(abs_diff):.6f}"
    )
    print(
        f"Rel err:  max={rel_err.max():.6f}, mean={rel_err.mean():.6f}, median={np.median(rel_err):.6f}"
    )

    corr = np.corrcoef(scores_np.flatten(), cpp_scores.flatten())[0, 1]
    print(f"Corr:     {corr:.6f}")

    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    print(
        f"Max diff at {max_idx}: PyTorch={scores_np[max_idx]:.6f}, C++={cpp_scores[max_idx]:.6f}"
    )

    if abs_diff.max() < 1e-4:
        print(f"✓ Scores MATCH")
    else:
        print(
            f"✗ Scores MISMATCH - Issue is in attention computation (RoPE, repeat_kv, transpose, or matmul)"
        )


if __name__ == "__main__":
    main()

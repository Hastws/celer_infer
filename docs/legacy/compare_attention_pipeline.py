#!/usr/bin/env python3
"""
Compare outputs at each stage: attn_out_flat -> attn_proj -> h1_attn_l0
"""

import numpy as np
import json
import os


def load_binary(path, shape):
    """Load raw float32 binary file."""
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)


def compare(name, torch_arr, cpp_arr):
    """Compare two arrays."""
    print(f"\n{name}:")
    if torch_arr.shape != cpp_arr.shape:
        print(f"  ✗ Shape mismatch: {torch_arr.shape} vs {cpp_arr.shape}")
        return False

    abs_diff = np.abs(torch_arr - cpp_arr)
    print(
        f"  PyTorch: min={torch_arr.min():.8f}, max={torch_arr.max():.8f}, mean={torch_arr.mean():.8f}"
    )
    print(
        f"  C++:     min={cpp_arr.min():.8f}, max={cpp_arr.max():.8f}, mean={cpp_arr.mean():.8f}"
    )
    print(f"  Max diff: {abs_diff.max():.8f}, Mean diff: {abs_diff.mean():.8f}")

    if abs_diff.max() < 1e-5:
        print(f"  ✓ MATCH")
        return True
    else:
        print(f"  ✗ MISMATCH")
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(
            f"    Max at {idx}: PyTorch={torch_arr[idx]:.8f}, C++={cpp_arr[idx]:.8f}, diff={abs_diff[idx]:.8f}"
        )
        return False


def main():
    dump_dir = "dump_minimind"

    with open(os.path.join(dump_dir, "minimind.json")) as f:
        cfg = json.load(f)

    B, S = cfg["meta"]["B"], cfg["meta"]["S"]
    hidden = cfg["config"]["hidden_size"]
    heads = cfg["config"]["num_attention_heads"]
    head_dim = hidden // heads

    print(f"Config: B={B}, S={S}, hidden={hidden}")

    # Load files
    torch_attn_out_flat = np.load(
        os.path.join(dump_dir, "attn_out_flat_torch.npy")
    )  # (B, S, H*D)
    cpp_attn_out_flat = load_binary(
        os.path.join(dump_dir, "attn_out_flat_l0_cpp.npy"), (B, S, heads * head_dim)
    )

    torch_attn_proj = np.load(
        os.path.join(dump_dir, "attn_proj_torch.npy")
    )  # (B, S, hidden)
    # C++ doesn't save attn_proj separately, so we need to compare h1_attn_l0 which includes residual
    torch_h1_attn = np.load(
        os.path.join(dump_dir, "h1_attn_l0_torch.npy")
    )  # (B, S, hidden)
    cpp_h1_attn = load_binary(
        os.path.join(dump_dir, "h1_attn_l0_cpp.npy"), (B, S, hidden)
    )

    torch_h0_emb = np.load(os.path.join(dump_dir, "h0_torch.npy"))  # (B, S, hidden)
    cpp_h0_emb = load_binary(os.path.join(dump_dir, "h0_cpp.npy"), (B, S, hidden))

    print(f"\n{'='*80}")
    print(f"ATTENTION OUTPUT PIPELINE COMPARISON")
    print(f"{'='*80}")

    m1 = compare(
        "1. Attention output (after merge heads)",
        torch_attn_out_flat,
        cpp_attn_out_flat,
    )

    # Can't directly compare attn_proj since C++ doesn't save it
    # But we can compute: torch_h1_attn_computed = attn_proj + h0_emb (residual)
    torch_h1_attn_computed = torch_attn_proj + torch_h0_emb
    m2 = compare(
        "2. Attention output (after attn_proj + residual)",
        torch_h1_attn_computed,
        cpp_h1_attn,
    )

    print(f"\n{'='*80}")
    if m1 and m2:
        print(f"✓ All stages MATCH - inference should be correct")
    elif m1:
        print(f"✓ attn_out_flat matches")
        print(f"✗ After residual, mismatches occur")
        print(f"  Issue might be in: w_o weights or residual addition formula")
    else:
        print(f"✗ attn_out_flat itself mismatches")


if __name__ == "__main__":
    main()

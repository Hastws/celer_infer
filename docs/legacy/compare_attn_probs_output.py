#!/usr/bin/env python3
"""
Compare attention prob and attention output between PyTorch and C++.
"""

import numpy as np
import os


def load_binary(path, shape):
    """Load raw float32 binary file."""
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)


def compare(name, torch_arr, cpp_arr):
    """Compare two arrays."""
    print(f"\n{name}:")
    print(f"  PyTorch shape: {torch_arr.shape}, C++ shape: {cpp_arr.shape}")

    if torch_arr.shape != cpp_arr.shape:
        print(f"  ✗ SHAPE MISMATCH")
        return False

    abs_diff = np.abs(torch_arr - cpp_arr)
    print(f"  Max diff: {abs_diff.max():.8f}")
    print(f"  Mean diff: {abs_diff.mean():.8f}")

    if abs_diff.max() < 1e-4:
        print(f"  ✓ MATCH")
        return True
    else:
        print(f"  ✗ MISMATCH")
        # Show which values differ
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(
            f"    Max diff at {idx}: PyTorch={torch_arr[idx]:.8f}, C++={cpp_arr[idx]:.8f}"
        )
        return False


def main():
    dump_dir = "dump_minimind"

    B, S = 2, 5
    heads = 8
    hidden = 64
    head_dim = 8

    # Load PyTorch arrays
    torch_probs = np.load(
        os.path.join(dump_dir, "scores_l0_torch_manual.npy")
    )  # (B, H, S, S)
    # Apply softmax
    from scipy.special import softmax

    torch_probs = softmax(torch_probs, axis=-1)

    torch_attn_out = np.load(
        os.path.join(dump_dir, "attn_out_flat_torch.npy")
    )  # (B, S, H*D)

    # Load C++ arrays
    cpp_probs = load_binary(
        os.path.join(dump_dir, "probs_l0_cpp.npy"), (B, heads, S, S)
    )
    cpp_attn_out = load_binary(
        os.path.join(dump_dir, "attn_out_flat_l0_cpp.npy"), (B, S, heads * head_dim)
    )

    print(f"\n{'='*80}")
    print(f"ATTENTION PROBS & OUTPUT COMPARISON")
    print(f"{'='*80}")

    probs_match = compare(
        "Attention Probabilities (after softmax)", torch_probs, cpp_probs
    )
    attn_match = compare(
        "Attention Output (after probs @ v)", torch_attn_out, cpp_attn_out
    )

    print(f"\n{'='*80}")
    if probs_match and attn_match:
        print(f"✓ Attention probs and output both MATCH")
        print(f"  Issue must be in: output projection (w_o) or residual connection")
    elif probs_match:
        print(f"✗ Attention probs match but attn_out_flat MISMATCHES")
        print(f"  Issue is in: probs @ v computation")
    else:
        print(f"✗ Attention probs MISMATCH")
        print(f"  Issue is in: softmax or masking")


if __name__ == "__main__":
    main()

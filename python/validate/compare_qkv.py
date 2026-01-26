#!/usr/bin/env python3
"""
Compare Q/K/V projections and attention scores between PyTorch and C++.
"""

import numpy as np
import os


def load_binary(path, shape):
    """Load raw float32 binary file."""
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)


def compute_stats(arr):
    """Compute statistics for an array."""
    return {
        "min": arr.min(),
        "max": arr.max(),
        "mean": arr.mean(),
        "std": arr.std(),
    }


def compare_arrays(name, torch_arr, cpp_arr):
    """Compare PyTorch and C++ arrays."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    if torch_arr.shape != cpp_arr.shape:
        print(f"✗ SHAPE MISMATCH: PyTorch {torch_arr.shape} vs C++ {cpp_arr.shape}")
        return

    torch_stats = compute_stats(torch_arr)
    cpp_stats = compute_stats(cpp_arr)

    abs_diff = np.abs(torch_arr - cpp_arr)
    rel_err = np.where(
        np.abs(torch_arr) > 1e-6,
        np.abs(torch_arr - cpp_arr) / (np.abs(torch_arr) + 1e-8),
        np.abs(torch_arr - cpp_arr),
    )

    corr = np.corrcoef(torch_arr.flatten(), cpp_arr.flatten())[0, 1]

    print(
        f"PyTorch:  min={torch_stats['min']:.6f}, max={torch_stats['max']:.6f}, mean={torch_stats['mean']:.6f}, std={torch_stats['std']:.6f}"
    )
    print(
        f"C++:      min={cpp_stats['min']:.6f}, max={cpp_stats['max']:.6f}, mean={cpp_stats['mean']:.6f}, std={cpp_stats['std']:.6f}"
    )
    print(
        f"Abs diff: max={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}, median={np.median(abs_diff):.6f}"
    )
    print(
        f"Rel err:  max={rel_err.max():.6f}, mean={rel_err.mean():.6f}, median={np.median(rel_err):.6f}"
    )
    print(f"Corr:     {corr:.6f}")

    # Find max diff location
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    print(
        f"Max diff at {max_idx}: PyTorch={torch_arr[max_idx]:.6f}, C++={cpp_arr[max_idx]:.6f}"
    )

    # Check match
    if abs_diff.max() < 1e-5:
        print(f"✓ MATCH (atol=1e-5)")
    else:
        print(f"✗ MISMATCH")

    return abs_diff.max() < 1e-5


def main():
    dump_dir = "dump_minimind"

    # Config from JSON
    import json

    with open(os.path.join(dump_dir, "minimind.json")) as f:
        config = json.load(f)

    B, S = config["meta"]["B"], config["meta"]["S"]
    hidden = config["config"]["hidden_size"]
    heads = config["config"]["num_attention_heads"]
    kv_heads = config["config"]["num_key_value_heads"]
    head_dim = hidden // heads

    print(
        f"Config: B={B}, S={S}, hidden={hidden}, heads={heads}, kv_heads={kv_heads}, head_dim={head_dim}"
    )
    print(
        f"Shapes: q_flat=(B,S,H*D)=({B},{S},{heads*head_dim}), k_flat=(B,S,KVH*D)=({B},{S},{kv_heads*head_dim}), v_flat=(B,S,KVH*D)=({B},{S},{kv_heads*head_dim})"
    )

    # Load PyTorch arrays
    torch_q = np.load(os.path.join(dump_dir, "q_flat_l0_torch.npy"))
    torch_k = np.load(os.path.join(dump_dir, "k_flat_l0_torch.npy"))
    torch_v = np.load(os.path.join(dump_dir, "v_flat_l0_torch.npy"))
    torch_h_norm = np.load(os.path.join(dump_dir, "h_norm_l0_torch.npy"))

    # Load C++ arrays (raw binary)
    cpp_q = load_binary(
        os.path.join(dump_dir, "q_flat_l0_cpp.npy"), (B, S, heads * head_dim)
    )
    cpp_k = load_binary(
        os.path.join(dump_dir, "k_flat_l0_cpp.npy"), (B, S, kv_heads * head_dim)
    )
    cpp_v = load_binary(
        os.path.join(dump_dir, "v_flat_l0_cpp.npy"), (B, S, kv_heads * head_dim)
    )
    cpp_h_norm = load_binary(
        os.path.join(dump_dir, "h_norm_l0_cpp.npy"), (B, S, hidden)
    )

    print("\n" + "=" * 80)
    print("LAYER 0 ATTENTION PROJECTION COMPARISON")
    print("=" * 80)

    # Compare h_norm
    compare_arrays("0. h_norm_l0 (After input_layernorm)", torch_h_norm, cpp_h_norm)

    # Compare Q, K, V projections
    q_match = compare_arrays("1. q_flat_l0 (Q projections)", torch_q, cpp_q)
    k_match = compare_arrays("2. k_flat_l0 (K projections)", torch_k, cpp_k)
    v_match = compare_arrays("3. v_flat_l0 (V projections)", torch_v, cpp_v)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if q_match:
        print("✓ Q projections MATCH")
    else:
        print("✗ Q projections MISMATCH")

    if k_match:
        print("✓ K projections MATCH")
    else:
        print("✗ K projections MISMATCH")

    if v_match:
        print("✓ V projections MATCH")
    else:
        print("✗ V projections MISMATCH")

    # Check if all match
    if q_match and k_match and v_match:
        print(
            "\n✓ All projections match! Issue is in post-projection attention ops (RoPE, masking, softmax, etc.)"
        )
    else:
        print("\n✗ Projection mismatch detected. Issue is in Q/K/V computation.")


if __name__ == "__main__":
    main()

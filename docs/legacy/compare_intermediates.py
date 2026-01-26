#!/usr/bin/env python3
"""
Compare layer 0 intermediate outputs between PyTorch and C++.
"""

import numpy as np
import os


def load_binary(path, shape):
    """Load raw binary float32 file."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


def compare_arrays(torch_arr, cpp_arr, name):
    """Compare two arrays and print stats."""
    if torch_arr.shape != cpp_arr.shape:
        print(f"\n❌ {name}: Shape mismatch!")
        print(f"   PyTorch shape: {torch_arr.shape}")
        print(f"   C++ shape: {cpp_arr.shape}")
        return

    diff = np.abs(torch_arr - cpp_arr)
    rel_error = np.abs((torch_arr - cpp_arr) / (np.abs(torch_arr) + 1e-8))
    correlation = np.corrcoef(torch_arr.flatten(), cpp_arr.flatten())[0, 1]

    print(f"\n{name}:")
    print(
        f"  PyTorch:  min={torch_arr.min():.6f}, max={torch_arr.max():.6f}, mean={torch_arr.mean():.6f}, std={torch_arr.std():.6f}"
    )
    print(
        f"  C++:      min={cpp_arr.min():.6f}, max={cpp_arr.max():.6f}, mean={cpp_arr.mean():.6f}, std={cpp_arr.std():.6f}"
    )
    print(
        f"  Abs diff: max={diff.max():.6f}, mean={diff.mean():.6f}, median={np.median(diff):.6f}"
    )
    print(
        f"  Rel err:  max={rel_error.max():.6f}, mean={rel_error.mean():.6f}, median={np.median(rel_error):.6f}"
    )
    print(f"  Corr:     {correlation:.6f}")

    if np.allclose(torch_arr, cpp_arr, atol=1e-5):
        print(f"  ✓ MATCH (atol=1e-5)")
        return True
    else:
        print(f"  ✗ MISMATCH")
        # Find most different element
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(
            f"  Max diff at {max_idx}: PyTorch={torch_arr[max_idx]:.6f}, C++={cpp_arr[max_idx]:.6f}"
        )
        return False


def main():
    dump_dir = "dump_minimind"
    shape = (2, 5, 64)

    print("=" * 80)
    print("LAYER 0 INTERMEDIATE COMPARISON: PyTorch vs C++")
    print("=" * 80)

    results = {}

    # 1. Embedding (h0)
    h0_torch = np.load(os.path.join(dump_dir, "h0_torch.npy"))
    h0_cpp = load_binary(os.path.join(dump_dir, "h0_cpp.npy"), shape)
    results["h0 (embedding)"] = compare_arrays(
        h0_torch, h0_cpp, "1. h0 (Embedding output)"
    )

    # 2. After input_layernorm (h_norm)
    h_norm_torch = np.load(os.path.join(dump_dir, "h_norm_l0_torch.npy"))
    h_norm_cpp = load_binary(os.path.join(dump_dir, "h_norm_l0_cpp.npy"), shape)
    results["h_norm_l0"] = compare_arrays(
        h_norm_torch, h_norm_cpp, "2. h_norm_l0 (After input_layernorm)"
    )

    # 3. After attention (h1_attn)
    h1_attn_torch = np.load(os.path.join(dump_dir, "h1_attn_l0_torch.npy"))
    h1_attn_cpp = load_binary(os.path.join(dump_dir, "h1_attn_l0_cpp.npy"), shape)
    results["h1_attn_l0"] = compare_arrays(
        h1_attn_torch, h1_attn_cpp, "3. h1_attn_l0 (After attention + residual)"
    )

    # 4. After FFN (h0_ffn)
    h0_ffn_torch = np.load(os.path.join(dump_dir, "h0_ffn_l0_torch.npy"))
    h0_ffn_cpp = load_binary(os.path.join(dump_dir, "h0_ffn_l0_cpp.npy"), shape)
    results["h0_ffn_l0"] = compare_arrays(
        h0_ffn_torch, h0_ffn_cpp, "4. h0_ffn_l0 (After FFN + residual)"
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    match_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, matched in results.items():
        status = "✓" if matched else "✗"
        print(f"{status} {name}")

    print(f"\nMatches: {match_count}/{total_count}")

    # Identify where divergence occurs
    print("\n" + "=" * 80)
    if results["h0 (embedding)"] and not results["h_norm_l0"]:
        print("⚠️  DIVERGENCE AT: input_layernorm (RMSNorm)")
    elif results["h_norm_l0"] and not results["h1_attn_l0"]:
        print("⚠️  DIVERGENCE AT: Attention computation")
    elif results["h1_attn_l0"] and not results["h0_ffn_l0"]:
        print("⚠️  DIVERGENCE AT: FFN computation")
    elif all(results.values()):
        print(
            "✓ All intermediates match! Issue must be in final layer or output projection."
        )
    else:
        print("❌ Multiple divergences detected")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare attn_proj between C++ and PyTorch.
"""

import numpy as np
import os

dump_dir = "dump_minimind"

# Load C++ attn_proj
attn_proj_cpp = np.fromfile(
    os.path.join(dump_dir, "attn_proj_l0_cpp.npy"), dtype=np.float32
)
attn_proj_cpp = attn_proj_cpp.reshape(2, 5, 64)

# Load PyTorch attn_proj
attn_proj_torch = np.load(os.path.join(dump_dir, "attn_proj_l0_torch.npy"))

print("attn_proj Comparison")
print("=" * 60)
print(f"Shape: {attn_proj_cpp.shape}")
print(f"\nC++ attn_proj stats:")
print(
    f"  min={attn_proj_cpp.min():.8f}, max={attn_proj_cpp.max():.8f}, mean={attn_proj_cpp.mean():.8f}"
)

print(f"\nPyTorch attn_proj stats:")
print(
    f"  min={attn_proj_torch.min():.8f}, max={attn_proj_torch.max():.8f}, mean={attn_proj_torch.mean():.8f}"
)

# Compare
diff = np.abs(attn_proj_cpp - attn_proj_torch)
print(f"\nDifference:")
print(f"  max={diff.max():.8e}, mean={diff.mean():.8e}, std={diff.std():.8e}")

# Find where mismatch is largest
max_idx = np.unravel_index(np.argmax(diff), diff.shape)
print(f"\nLargest difference at {max_idx}:")
print(f"  C++: {attn_proj_cpp[max_idx]:.8f}")
print(f"  PyTorch: {attn_proj_torch[max_idx]:.8f}")
print(f"  Diff: {diff[max_idx]:.8e}")

# Check values at [0,0,:10]
print(f"\nFirst 10 values at [0,0,:]:")
print(f"  C++: {attn_proj_cpp[0,0,:10]}")
print(f"  PyTorch: {attn_proj_torch[0,0,:10]}")
print(f"  Diff: {diff[0,0,:10]}")

# Compare h1_attn (after residual)
print("\n" + "=" * 60)
print("h1_attn Comparison (after residual)")
print("=" * 60)

h1_attn_cpp = np.fromfile(
    os.path.join(dump_dir, "h1_attn_l0_cpp.npy"), dtype=np.float32
)
h1_attn_cpp = h1_attn_cpp.reshape(2, 5, 64)

h1_attn_torch = np.load(os.path.join(dump_dir, "h1_attn_l0_torch.npy"))

print(f"Shape: {h1_attn_cpp.shape}")
print(f"\nC++ h1_attn stats:")
print(
    f"  min={h1_attn_cpp.min():.8f}, max={h1_attn_cpp.max():.8f}, mean={h1_attn_cpp.mean():.8f}"
)

print(f"\nPyTorch h1_attn stats:")
print(
    f"  min={h1_attn_torch.min():.8f}, max={h1_attn_torch.max():.8f}, mean={h1_attn_torch.mean():.8f}"
)

# Compare
diff_h1 = np.abs(h1_attn_cpp - h1_attn_torch)
print(f"\nDifference:")
print(f"  max={diff_h1.max():.8e}, mean={diff_h1.mean():.8e}, std={diff_h1.std():.8e}")

if diff.max() < 1e-5:
    print("\n✓ attn_proj MATCHES!")
else:
    print("\n✗ attn_proj MISMATCHES!")

if diff_h1.max() < 1e-5:
    print("✓ h1_attn MATCHES!")
else:
    print("✗ h1_attn MISMATCHES!")

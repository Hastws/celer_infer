#!/usr/bin/env python3
"""
Debug the residual connection issue.
attn_proj matches, h1_attn doesn't = problem is in: h0 + attn_proj
"""

import numpy as np
import os

dump_dir = "dump_minimind"

# Load h0
h0_cpp = np.fromfile(os.path.join(dump_dir, "h0_cpp.npy"), dtype=np.float32).reshape(
    2, 5, 64
)
h0_torch = np.load(os.path.join(dump_dir, "h0_torch.npy"))

print("h0 (embedding) Comparison")
print("=" * 60)
print(f"\nC++ h0 stats:")
print(f"  min={h0_cpp.min():.8f}, max={h0_cpp.max():.8f}, mean={h0_cpp.mean():.8f}")

print(f"\nPyTorch h0 stats:")
print(
    f"  min={h0_torch.min():.8f}, max={h0_torch.max():.8f}, mean={h0_torch.mean():.8f}"
)

# Compare
diff = np.abs(h0_cpp - h0_torch)
print(f"\nDifference:")
print(f"  max={diff.max():.8e}, mean={diff.mean():.8e}")

print(f"\nFirst 10 values at [0,0,:]:")
print(f"  C++: {h0_cpp[0,0,:10]}")
print(f"  PyTorch: {h0_torch[0,0,:10]}")

# Load attn_proj
attn_proj_cpp = np.fromfile(
    os.path.join(dump_dir, "attn_proj_l0_cpp.npy"), dtype=np.float32
).reshape(2, 5, 64)
attn_proj_torch = np.load(os.path.join(dump_dir, "attn_proj_l0_torch.npy"))

print("\n" + "=" * 60)
print("Residual Addition Analysis")
print("=" * 60)

# Manual residual computation
h1_manual = h0_cpp + attn_proj_cpp
h1_torch = h0_torch + attn_proj_torch

# Load actual h1_attn from C++
h1_attn_cpp = np.fromfile(
    os.path.join(dump_dir, "h1_attn_l0_cpp.npy"), dtype=np.float32
).reshape(2, 5, 64)
h1_attn_torch = np.load(os.path.join(dump_dir, "h1_attn_l0_torch.npy"))

print("\nManual PyTorch h0 + attn_proj:")
print(
    f"  min={h1_torch.min():.8f}, max={h1_torch.max():.8f}, mean={h1_torch.mean():.8f}"
)

print("\nManual C++ h0 + attn_proj:")
print(
    f"  min={h1_manual.min():.8f}, max={h1_manual.max():.8f}, mean={h1_manual.mean():.8f}"
)

print("\nActual C++ h1_attn:")
print(
    f"  min={h1_attn_cpp.min():.8f}, max={h1_attn_cpp.max():.8f}, mean={h1_attn_cpp.mean():.8f}"
)

# Compare manual C++ (h0 + attn_proj) vs actual C++ h1_attn
diff_cpp = np.abs(h1_manual - h1_attn_cpp)
print(f"\nDifference between C++ manual (h0+attn_proj) and actual C++ h1_attn:")
print(f"  max={diff_cpp.max():.8e}, mean={diff_cpp.mean():.8e}")

if diff_cpp.max() < 1e-5:
    print("  ✓ C++ residual is correct!")
else:
    print("  ✗ C++ residual is WRONG!")

print(f"\nFirst values at [0,0,:]:")
print(f"  h0_cpp: {h0_cpp[0,0,:10]}")
print(f"  attn_proj_cpp: {attn_proj_cpp[0,0,:10]}")
print(f"  Manual h0+proj: {h1_manual[0,0,:10]}")
print(f"  Actual h1_cpp: {h1_attn_cpp[0,0,:10]}")
print(f"  PyTorch h1: {h1_torch[0,0,:10]}")

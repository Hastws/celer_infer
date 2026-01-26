#!/usr/bin/env python3
"""
Compare final logits between C++ and PyTorch.
"""

import numpy as np
import os
import json

dump_dir = "dump_minimind"

# Load config
with open(os.path.join(dump_dir, "minimind.json")) as f:
    cfg = json.load(f)

B = cfg["meta"]["B"]
S = cfg["meta"]["S"]
vocab = cfg["config"]["vocab_size"]

# Load logits as binary files
logits_cpp = np.fromfile(
    os.path.join(dump_dir, "logits_cpp.npy"), dtype=np.float32
).reshape(B, S, vocab)
logits_torch = np.fromfile(
    os.path.join(dump_dir, "logits_torch.npy"), dtype=np.float32
).reshape(B, S, vocab)

print("Final Logits Comparison")
print("=" * 60)
print(f"Shape: {logits_cpp.shape}")

print(f"\nC++ logits stats:")
print(
    f"  min={logits_cpp.min():.8f}, max={logits_cpp.max():.8f}, mean={logits_cpp.mean():.8f}"
)

print(f"\nPyTorch logits stats:")
print(
    f"  min={logits_torch.min():.8f}, max={logits_torch.max():.8f}, mean={logits_torch.mean():.8f}"
)

# Compare
diff = np.abs(logits_cpp - logits_torch)
print(f"\nDifference:")
print(f"  max={diff.max():.8e}, mean={diff.mean():.8e}, std={diff.std():.8e}")

# Find where mismatch is largest
max_idx = np.unravel_index(np.argmax(diff), diff.shape)
print(f"\nLargest difference at {max_idx}:")
print(f"  C++: {logits_cpp[max_idx]:.8f}")
print(f"  PyTorch: {logits_torch[max_idx]:.8f}")
print(f"  Diff: {diff[max_idx]:.8e}")

# Check correlation
correlation = np.corrcoef(logits_cpp.flatten(), logits_torch.flatten())[0, 1]
print(f"\nCorrelation: {correlation:.8f}")

if diff.max() < 1e-5:
    print("\n✓ Logits MATCH!")
else:
    print("\n✗ Logits MISMATCH!")

print(f"\nFirst 20 logits at [0,0,:]:")
print(f"  C++: {logits_cpp[0,0,:20]}")
print(f"  PyTorch: {logits_torch[0,0,:20]}")

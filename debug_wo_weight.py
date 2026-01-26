#!/usr/bin/env python3
"""
Debug output projection weight comparison.
"""

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
        cfg = json.load(f)

    B, S = cfg["meta"]["B"], cfg["meta"]["S"]
    hidden = cfg["config"]["hidden_size"]
    heads = cfg["config"]["num_attention_heads"]
    head_dim = hidden // heads

    # Load w_o weight from layer 0
    layer0 = cfg["weights"]["layers"][0]
    wo = load_tensor_from_json(layer0["wo"])  # (hidden, H*D)

    print(f"wo shape: {wo.shape}")
    print(f"wo dtype: {wo.dtype}")
    print(
        f"wo stats: min={wo.min():.8f}, max={wo.max():.8f}, mean={wo.mean():.8f}, std={wo.std():.8f}"
    )

    # Load attn_out_flat
    attn_out_flat = np.load(
        os.path.join(dump_dir, "attn_out_flat_torch.npy")
    )  # (B, S, H*D)

    print(f"\nattn_out_flat shape: {attn_out_flat.shape}")
    print(f"attn_out_flat[0,0,:10]: {attn_out_flat[0, 0, :10]}")

    # Compute attn_proj manually: (B, S, H*D) @ (H*D, hidden)^T
    # In PyTorch: attn_out @ wo.t()
    # But since wo is (hidden, H*D), wo.t() is (H*D, hidden)
    # So we need: attn_out_flat (B,S,H*D) @ wo.t() (H*D, hidden) -> (B,S,hidden)
    attn_proj = np.dot(attn_out_flat, wo.T)  # (B, S, hidden)

    print(f"\nComputed attn_proj shape: {attn_proj.shape}")
    print(
        f"attn_proj stats: min={attn_proj.min():.8f}, max={attn_proj.max():.8f}, mean={attn_proj.mean():.8f}"
    )

    # Load h0 (embedding) for residual
    h0_emb = np.load(os.path.join(dump_dir, "h0_torch.npy"))

    # Compute h1_attn = attn_proj + h0_emb (residual)
    h1_attn = attn_proj + h0_emb

    print(
        f"\nComputed h1_attn (proj + residual) stats: min={h1_attn.min():.8f}, max={h1_attn.max():.8f}, mean={h1_attn.mean():.8f}"
    )

    # Load saved attn_proj
    saved_attn_proj = np.load(os.path.join(dump_dir, "attn_proj_torch.npy"))
    print(
        f"\nSaved attn_proj stats: min={saved_attn_proj.min():.8f}, max={saved_attn_proj.max():.8f}, mean={saved_attn_proj.mean():.8f}"
    )

    # Compare
    diff = np.abs(attn_proj - saved_attn_proj)
    print(f"Difference between computed and saved attn_proj:")
    print(f"  Max: {diff.max():.8e}, Mean: {diff.mean():.8e}")

    # Load C++ w_o
    # C++ loads wo as (hidden, H*D) and does matmul_nt which is: A @ B^T
    # where A is (B,S,H*D) and B is (hidden, H*D)
    # So: (B,S,H*D) @ (hidden, H*D)^T = (B,S,H*D) @ (H*D, hidden) = (B,S,hidden)
    # Which is the same as PyTorch

    print(f"\nKey question: Does C++ apply w_o correctly?")
    print(f"C++ matmul_nt(attn_out_flat, B*S, H*D, w_o, hidden, H*D, out, B*S, hidden)")
    print(f"  = (B*S, H*D) @ (hidden, H*D)^T")
    print(f"  = (B*S, H*D) @ (H*D, hidden)")
    print(f"  = (B*S, hidden)")

    # Load C++ attn_out_flat
    cpp_attn_out_flat = np.fromfile(
        os.path.join(dump_dir, "attn_out_flat_l0_cpp.npy"), dtype=np.float32
    )
    cpp_attn_out_flat = cpp_attn_out_flat.reshape(B, S, heads * head_dim)

    # Compute what C++ should compute
    cpp_attn_proj = np.dot(cpp_attn_out_flat, wo.T)
    cpp_h1_attn = cpp_attn_proj + np.fromfile(
        os.path.join(dump_dir, "h0_cpp.npy"), dtype=np.float32
    ).reshape(B, S, hidden)

    print(
        f"\nPredicted C++ h1_attn stats: min={cpp_h1_attn.min():.8f}, max={cpp_h1_attn.max():.8f}, mean={cpp_h1_attn.mean():.8f}"
    )

    actual_cpp_h1_attn = np.fromfile(
        os.path.join(dump_dir, "h1_attn_l0_cpp.npy"), dtype=np.float32
    ).reshape(B, S, hidden)
    print(
        f"Actual C++ h1_attn stats: min={actual_cpp_h1_attn.min():.8f}, max={actual_cpp_h1_attn.max():.8f}, mean={actual_cpp_h1_attn.mean():.8f}"
    )

    diff = np.abs(cpp_h1_attn - actual_cpp_h1_attn)
    print(f"Difference: max={diff.max():.8f}, mean={diff.mean():.8f}")


if __name__ == "__main__":
    main()

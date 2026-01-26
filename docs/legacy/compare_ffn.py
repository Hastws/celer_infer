#!/usr/bin/env python3
"""
Compare layer 0 FFN output between C++ and PyTorch.
"""

import numpy as np
import torch
import json
import os
from script.llm_minimind_model import MiniMindModel, MiniMindConfig
import base64


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


dump_dir = "dump_minimind"

# Load config
with open(os.path.join(dump_dir, "minimind.json")) as f:
    cfg_json = json.load(f)

B = cfg_json["meta"]["B"]
S = cfg_json["meta"]["S"]
hidden = cfg_json["config"]["hidden_size"]
heads = cfg_json["config"]["num_attention_heads"]

# Load C++ intermediate outputs
h1_attn_cpp = np.fromfile(
    os.path.join(dump_dir, "h1_attn_l0_cpp.npy"), dtype=np.float32
).reshape(B, S, hidden)
h0_ffn_cpp = np.fromfile(
    os.path.join(dump_dir, "h0_ffn_l0_cpp.npy"), dtype=np.float32
).reshape(B, S, hidden)

# Create model and load weights
model_config = MiniMindConfig(
    vocab_size=cfg_json["config"]["vocab_size"],
    hidden_size=cfg_json["config"]["hidden_size"],
    num_hidden_layers=cfg_json["config"]["num_hidden_layers"],
    num_attention_heads=cfg_json["config"]["num_attention_heads"],
    num_key_value_heads=cfg_json["config"].get(
        "num_key_value_heads", cfg_json["config"]["num_attention_heads"]
    ),
    intermediate_size=cfg_json["config"]["intermediate_size"],
    max_position_embeddings=cfg_json["config"]["max_position_embeddings"],
)

model = MiniMindModel(model_config)
model.eval()

# Load weights
weights = cfg_json["weights"]

# Load layer 0 weights
layer0 = model.layers[0]
layer_json = weights["layers"][0]

# Load FFN weights
w_gate = load_tensor_from_json(layer_json["w_gate"])
w_up = load_tensor_from_json(layer_json["w_up"])
w_down = load_tensor_from_json(layer_json["w_down"])

with torch.no_grad():
    layer0.mlp.gate_proj.weight.copy_(torch.tensor(w_gate, dtype=torch.float32))
    layer0.mlp.up_proj.weight.copy_(torch.tensor(w_up, dtype=torch.float32))
    layer0.mlp.down_proj.weight.copy_(torch.tensor(w_down, dtype=torch.float32))

# Compute FFN from h1_attn (which we know matches)
h1_attn_torch = torch.tensor(h1_attn_cpp, dtype=torch.float32)

with torch.no_grad():
    # Gate and Up
    ffn_gate = torch.matmul(
        h1_attn_torch, layer0.mlp.gate_proj.weight.t()
    )  # (B, S, inter)
    ffn_up = torch.matmul(h1_attn_torch, layer0.mlp.up_proj.weight.t())  # (B, S, inter)

    # Apply SiLU activation and multiply
    ffn_mid = torch.nn.functional.silu(ffn_gate) * ffn_up  # (B, S, inter)

    # Down projection
    h0_ffn_torch = (
        torch.matmul(ffn_mid, layer0.mlp.down_proj.weight.t()) + h1_attn_torch
    )  # Residual

    print("FFN Layer 0 Output Comparison")
    print("=" * 60)
    print(f"Shape: {h0_ffn_torch.shape}\n")

    print(
        f"C++ h0_ffn:  min={h0_ffn_cpp.min():.8f}, max={h0_ffn_cpp.max():.8f}, mean={h0_ffn_cpp.mean():.8f}"
    )
    print(
        f"PyTorch:     min={h0_ffn_torch.min():.8f}, max={h0_ffn_torch.max():.8f}, mean={h0_ffn_torch.mean():.8f}"
    )

    h0_ffn_np = h0_ffn_torch.numpy()
    diff = np.abs(h0_ffn_cpp - h0_ffn_np)
    print(f"\nDifference: max={diff.max():.8e}, mean={diff.mean():.8e}")

    if diff.max() < 1e-5:
        print("✓ FFN output MATCHES!")
    else:
        print("✗ FFN output MISMATCHES!")

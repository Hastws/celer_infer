#!/usr/bin/env python3
"""
Debug FFN layer 0 comparison.
"""

import numpy as np
import torch
import torch.nn.functional as F
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
inter = cfg_json["config"]["intermediate_size"]

# Load C++ intermediate outputs
h1_attn_cpp = np.fromfile(
    os.path.join(dump_dir, "h1_attn_l0_cpp.npy"), dtype=np.float32
).reshape(B, S, hidden)
h0_ffn_cpp = np.fromfile(
    os.path.join(dump_dir, "h0_ffn_l0_cpp.npy"), dtype=np.float32
).reshape(B, S, hidden)

print(
    f"h1_attn_cpp shape: {h1_attn_cpp.shape}, range: [{h1_attn_cpp.min():.6f}, {h1_attn_cpp.max():.6f}]"
)
print(
    f"h0_ffn_cpp shape: {h0_ffn_cpp.shape}, range: [{h0_ffn_cpp.min():.6f}, {h0_ffn_cpp.max():.6f}]"
)

# Create model
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
layer0 = model.layers[0]
layer_json = weights["layers"][0]

# Load FFN weights
w_gate = load_tensor_from_json(layer_json["w_gate"])
w_up = load_tensor_from_json(layer_json["w_up"])
w_down = load_tensor_from_json(layer_json["w_down"])

print(f"\nFFN weights loaded:")
print(
    f"  w_gate shape: {w_gate.shape}, range: [{w_gate.min():.6f}, {w_gate.max():.6f}]"
)
print(f"  w_up shape: {w_up.shape}, range: [{w_up.min():.6f}, {w_up.max():.6f}]")
print(
    f"  w_down shape: {w_down.shape}, range: [{w_down.min():.6f}, {w_down.max():.6f}]"
)

# Load RMSNorm weights
w_norm_post = load_tensor_from_json(layer_json["rms_ffn"])

print(f"RMSNorm weight loaded:")
print(
    f"  w_norm_post shape: {w_norm_post.shape}, range: [{w_norm_post.min():.6f}, {w_norm_post.max():.6f}]"
)

with torch.no_grad():
    layer0.post_attention_layernorm.weight.copy_(
        torch.tensor(w_norm_post, dtype=torch.float32)
    )
    layer0.mlp.gate_proj.weight.copy_(torch.tensor(w_gate, dtype=torch.float32))
    layer0.mlp.up_proj.weight.copy_(torch.tensor(w_up, dtype=torch.float32))
    layer0.mlp.down_proj.weight.copy_(torch.tensor(w_down, dtype=torch.float32))

    print(f"  layer0.mlp.gate_proj.weight shape: {layer0.mlp.gate_proj.weight.shape}")
    print(f"  layer0.mlp.up_proj.weight shape: {layer0.mlp.up_proj.weight.shape}")
    print(f"  layer0.mlp.down_proj.weight shape: {layer0.mlp.down_proj.weight.shape}")

# Compute FFN from h1_attn
h1_attn_torch = torch.tensor(h1_attn_cpp, dtype=torch.float32)

print(f"\nComputing FFN...")
with torch.no_grad():
    # First apply post-attention RMSNorm
    h1_norm = layer0.post_attention_layernorm(h1_attn_torch)
    print(
        f"  h1_norm (after RMSNorm): range: [{h1_norm.min():.6f}, {h1_norm.max():.6f}]"
    )
    # First apply post-attention RMSNorm
    h1_norm = layer0.post_attention_layernorm(h1_attn_torch)
    print(
        f"  h1_norm (after RMSNorm): range: [{h1_norm.min():.6f}, {h1_norm.max():.6f}]"
    )

    # Gate projection
    ffn_gate = torch.matmul(h1_norm, layer0.mlp.gate_proj.weight.t())
    print(
        f"  ffn_gate shape: {ffn_gate.shape}, range: [{ffn_gate.min():.6f}, {ffn_gate.max():.6f}]"
    )

    # Up projection
    ffn_up = torch.matmul(h1_norm, layer0.mlp.up_proj.weight.t())
    print(
        f"  ffn_up shape: {ffn_up.shape}, range: [{ffn_up.min():.6f}, {ffn_up.max():.6f}]"
    )

    # SiLU activation
    ffn_gate_silu = F.silu(ffn_gate)
    print(
        f"  ffn_gate after SiLU: range: [{ffn_gate_silu.min():.6f}, {ffn_gate_silu.max():.6f}]"
    )

    # Multiply
    ffn_mid = ffn_gate_silu * ffn_up
    print(f"  ffn_mid (gate*up): range: [{ffn_mid.min():.6f}, {ffn_mid.max():.6f}]")

    # Down projection
    ffn_down_out = torch.matmul(ffn_mid, layer0.mlp.down_proj.weight.t())
    print(
        f"  ffn_down_out (before residual): range: [{ffn_down_out.min():.6f}, {ffn_down_out.max():.6f}]"
    )

    # Add residual
    h0_ffn_torch = ffn_down_out + h1_attn_torch
    print(
        f"  h0_ffn_torch (after residual): range: [{h0_ffn_torch.min():.6f}, {h0_ffn_torch.max():.6f}]"
    )

    h0_ffn_np = h0_ffn_torch.numpy()
    diff = np.abs(h0_ffn_cpp - h0_ffn_np)
    print(f"\nDifference: max={diff.max():.8e}, mean={diff.mean():.8e}")

#!/usr/bin/env python3
"""
Extract attention projection from PyTorch model - attn_out_flat through w_o projection.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
import os
import json
import base64

# Load model
from script.llm_minimind_model import MiniMindModel, MiniMindConfig


def b64_decode(b64_str):
    return base64.b64decode(b64_str)


def load_tensor_from_json(data_dict):
    """Load tensor from JSON dict with Base64-encoded data."""
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


# Load config from JSON
dump_dir = "dump_minimind"
with open(os.path.join(dump_dir, "minimind.json")) as f:
    cfg_json = json.load(f)

B = cfg_json["meta"]["B"]
S = cfg_json["meta"]["S"]
hidden = cfg_json["config"]["hidden_size"]
heads = cfg_json["config"]["num_attention_heads"]
head_dim = hidden // heads

# Load from JSON for ground truth
print(f"Loading config: B={B}, S={S}, hidden={hidden}, heads={heads}")

# Create config for model
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
    rope_theta=cfg_json["config"].get("rope_theta", 10000.0),
    rms_norm_eps=cfg_json["config"].get("rms_norm_eps", 1e-6),
)

model = MiniMindModel(model_config)
model.eval()

# Load weights into model from JSON
print("\nLoading weights from JSON...")
weights = cfg_json["weights"]

# Load embedding
tok_emb_data = load_tensor_from_json(weights["tok_embedding"])
with torch.no_grad():
    model.embed_tokens.weight.copy_(torch.tensor(tok_emb_data, dtype=torch.float32))

# Load layers
for layer_idx in range(model_config.num_hidden_layers):
    layer_json = weights["layers"][layer_idx]
    layer = model.layers[layer_idx]

    # Attention weights
    w_q = load_tensor_from_json(layer_json["wq"])
    w_k = load_tensor_from_json(layer_json["wk"])
    w_v = load_tensor_from_json(layer_json["wv"])
    w_o = load_tensor_from_json(layer_json["wo"])

    with torch.no_grad():
        layer.self_attn.q_proj.weight.copy_(torch.tensor(w_q, dtype=torch.float32))
        layer.self_attn.k_proj.weight.copy_(torch.tensor(w_k, dtype=torch.float32))
        layer.self_attn.v_proj.weight.copy_(torch.tensor(w_v, dtype=torch.float32))
        layer.self_attn.o_proj.weight.copy_(torch.tensor(w_o, dtype=torch.float32))

    # RMSNorm
    w_norm = load_tensor_from_json(layer_json["rms_attn"])
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(torch.tensor(w_norm, dtype=torch.float32))

    # FFN
    w_gate = load_tensor_from_json(layer_json["w_gate"])
    w_up = load_tensor_from_json(layer_json["w_up"])
    w_down = load_tensor_from_json(layer_json["w_down"])

    with torch.no_grad():
        layer.mlp.gate_proj.weight.copy_(torch.tensor(w_gate, dtype=torch.float32))
        layer.mlp.up_proj.weight.copy_(torch.tensor(w_up, dtype=torch.float32))
        layer.mlp.down_proj.weight.copy_(torch.tensor(w_down, dtype=torch.float32))

    # Post-attention norm
    w_norm_post = load_tensor_from_json(layer_json["rms_ffn"])
    with torch.no_grad():
        layer.post_attention_layernorm.weight.copy_(
            torch.tensor(w_norm_post, dtype=torch.float32)
        )

# Load input - use dummy input based on B and S
# We'll load h0_emb from C++ instead
input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # Dummy, won't be used
print(f"\nUsing dummy input - will load h0_emb from saved values")

# Forward pass - load h0 from C++ (embedding output)
with torch.no_grad():
    # Load h0_emb from C++ - binary file, not .npy
    h0_emb_bytes = np.fromfile(
        os.path.join(dump_dir, "h0_cpp.npy"), dtype=np.float32
    )  # (B*S*hidden,)
    h0_emb = torch.tensor(h0_emb_bytes.reshape(B, S, hidden), dtype=torch.float32)

    print(f"h0_emb shape: {h0_emb.shape}")
    np.save(os.path.join(dump_dir, "h0_torch.npy"), h0_emb.numpy())

    # Layer 0 processing
    layer0 = model.layers[0]

    # RMSNorm
    h_norm = layer0.input_layernorm(h0_emb)
    print(f"h_norm shape: {h_norm.shape}")
    np.save(os.path.join(dump_dir, "h_norm_l0_torch.npy"), h_norm.numpy())

    # Get attention weights
    wo_weight = layer0.self_attn.o_proj.weight  # (hidden, hidden)

    # Load attn_out_flat from C++ (which we verified matches PyTorch)
    attn_out_flat_cpp_bytes = np.fromfile(
        os.path.join(dump_dir, "attn_out_flat_l0_cpp.npy"), dtype=np.float32
    )
    attn_out_flat_cpp = attn_out_flat_cpp_bytes.reshape(B, S, heads * head_dim)
    attn_out_flat_torch = torch.tensor(attn_out_flat_cpp, dtype=torch.float32)

    print(f"\nattn_out_flat shape: {attn_out_flat_torch.shape}")

    # Apply o_proj: (B, S, H*D) @ (H*D, hidden)^T = (B, S, hidden)
    # PyTorch does: input @ weight.t() + bias
    # which is: (B,S,H*D) @ (hidden, H*D) = (B,S,hidden)
    attn_proj = torch.matmul(attn_out_flat_torch, wo_weight.t())

    print(f"attn_proj shape: {attn_proj.shape}")
    print(
        f"attn_proj stats: min={attn_proj.min():.8f}, max={attn_proj.max():.8f}, mean={attn_proj.mean():.8f}"
    )

    np.save(os.path.join(dump_dir, "attn_proj_l0_torch.npy"), attn_proj.numpy())

    # Add residual: attn_output + h0_emb
    h1_attn = attn_proj + h0_emb

    print(
        f"\nh1_attn (after residual) stats: min={h1_attn.min():.8f}, max={h1_attn.max():.8f}, mean={h1_attn.mean():.8f}"
    )
    np.save(os.path.join(dump_dir, "h1_attn_l0_torch.npy"), h1_attn.numpy())

    print("\nSaved PyTorch intermediate outputs:")
    print("  - h0_torch.npy")
    print("  - h_norm_l0_torch.npy")
    print("  - attn_proj_l0_torch.npy")
    print("  - h1_attn_l0_torch.npy")

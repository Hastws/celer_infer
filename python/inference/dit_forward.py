#!/usr/bin/env python3
"""
DiT (Diffusion Transformer) PyTorch 推理与验证

加载导出的 JSON 模型，运行 PyTorch 前向传播，并保存输出用于与 C++ 对比。

用法:
    python -m python.inference.dit_forward [--json-path dump_dit/dit.json]
    
环境变量:
    JSON_PATH: JSON 模型路径 (默认: dump_dit/dit.json)
    DUMP_DIR: 输出目录 (默认: dump_dit)
    WARMUP: 预热次数 (默认: 1)
"""

import os
import sys
import json
import base64
import time
from typing import Any, Dict

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from python.core.dit import ModelConfig, Diffusion_Planner


def _b64_decode_to_numpy(t: Dict[str, Any]) -> np.ndarray:
    """从 base64 解码为 numpy 数组"""
    assert t["encoding"] == "base64"
    raw = base64.b64decode(t["data"])
    dtype = np.dtype(t["dtype"])
    shape = tuple(int(x) for x in t["shape"])
    arr = np.frombuffer(raw, dtype=dtype).copy()
    return arr.reshape(shape)


def _load_json(path: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_cfg(cfg_j: Dict[str, Any], device: str = 'cpu') -> ModelConfig:
    """从 JSON 配置构建 ModelConfig"""
    cfg = ModelConfig(
        future_len=int(cfg_j["future_len"]),
        time_len=int(cfg_j["time_len"]),
        agent_state_dim=int(cfg_j["agent_state_dim"]),
        agent_num=int(cfg_j["agent_num"]),
        static_objects_state_dim=int(cfg_j["static_objects_state_dim"]),
        static_objects_num=int(cfg_j["static_objects_num"]),
        lane_len=int(cfg_j["lane_len"]),
        lane_state_dim=int(cfg_j["lane_state_dim"]),
        lane_num=int(cfg_j["lane_num"]),
        route_len=int(cfg_j["route_len"]),
        route_state_dim=int(cfg_j["route_state_dim"]),
        route_num=int(cfg_j["route_num"]),
        encoder_depth=int(cfg_j["encoder_depth"]),
        decoder_depth=int(cfg_j["decoder_depth"]),
        num_heads=int(cfg_j["num_heads"]),
        hidden_dim=int(cfg_j["hidden_dim"]),
        predicted_neighbor_num=int(cfg_j["predicted_neighbor_num"]),
        diffusion_model_type=str(cfg_j["diffusion_model_type"]),
        device=device,
    )
    return cfg


def _load_linear(linear: torch.nn.Linear, w: Dict[str, Any]) -> None:
    """加载 Linear 层权重"""
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(
            _b64_decode_to_numpy(w["weight"]).astype(np.float32)
        ))
        if "bias" in w and linear.bias is not None:
            linear.bias.copy_(torch.from_numpy(
                _b64_decode_to_numpy(w["bias"]).astype(np.float32)
            ))


def _load_layernorm(ln: torch.nn.LayerNorm, w: Dict[str, Any]) -> None:
    """加载 LayerNorm 层权重"""
    with torch.no_grad():
        ln.weight.copy_(torch.from_numpy(
            _b64_decode_to_numpy(w["weight"]).astype(np.float32)
        ))
        ln.bias.copy_(torch.from_numpy(
            _b64_decode_to_numpy(w["bias"]).astype(np.float32)
        ))


def _load_embedding(emb: torch.nn.Embedding, w: Dict[str, Any]) -> None:
    """加载 Embedding 层权重"""
    with torch.no_grad():
        emb.weight.copy_(torch.from_numpy(
            _b64_decode_to_numpy(w["weight"]).astype(np.float32)
        ))


def _load_mlp(mlp, w: Dict[str, Any]) -> None:
    """加载 timm Mlp 层权重"""
    _load_linear(mlp.fc1, w["fc1"])
    _load_linear(mlp.fc2, w["fc2"])


def _load_multihead_attention(mha: torch.nn.MultiheadAttention, w: Dict[str, Any]) -> None:
    """加载 MultiheadAttention 层权重"""
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.from_numpy(
            _b64_decode_to_numpy(w["in_proj_weight"]).astype(np.float32)
        ))
        mha.in_proj_bias.copy_(torch.from_numpy(
            _b64_decode_to_numpy(w["in_proj_bias"]).astype(np.float32)
        ))
        _load_linear(mha.out_proj, w["out_proj"])


def load_dit_weights(model: Diffusion_Planner, weights: Dict[str, Any]) -> None:
    """加载完整的 DiT 模型权重"""
    
    # ========== Encoder ==========
    encoder = model.encoder.encoder
    
    # Position embedding
    _load_linear(encoder.pos_emb, weights["encoder_pos_emb"])
    
    # Neighbor encoder
    neighbor_enc = encoder.neighbor_encoder
    w_neighbor = weights["neighbor_encoder"]
    _load_linear(neighbor_enc.type_emb, w_neighbor["type_emb"])
    _load_mlp(neighbor_enc.channel_pre_project, w_neighbor["channel_pre_project"])
    _load_mlp(neighbor_enc.token_pre_project, w_neighbor["token_pre_project"])
    for i, block in enumerate(neighbor_enc.blocks):
        w_block = w_neighbor["blocks"][i]
        _load_layernorm(block.norm1, w_block["norm1"])
        _load_mlp(block.channels_mlp, w_block["channels_mlp"])
        _load_layernorm(block.norm2, w_block["norm2"])
        _load_mlp(block.tokens_mlp, w_block["tokens_mlp"])
    _load_layernorm(neighbor_enc.norm, w_neighbor["norm"])
    _load_mlp(neighbor_enc.emb_project, w_neighbor["emb_project"])
    
    # Static encoder
    static_enc = encoder.static_encoder
    _load_mlp(static_enc.projection, weights["static_encoder"]["projection"])
    
    # Lane encoder
    lane_enc = encoder.lane_encoder
    w_lane = weights["lane_encoder"]
    _load_linear(lane_enc.speed_limit_emb, w_lane["speed_limit_emb"])
    _load_embedding(lane_enc.unknown_speed_emb, w_lane["unknown_speed_emb"])
    _load_linear(lane_enc.traffic_emb, w_lane["traffic_emb"])
    _load_mlp(lane_enc.channel_pre_project, w_lane["channel_pre_project"])
    _load_mlp(lane_enc.token_pre_project, w_lane["token_pre_project"])
    for i, block in enumerate(lane_enc.blocks):
        w_block = w_lane["blocks"][i]
        _load_layernorm(block.norm1, w_block["norm1"])
        _load_mlp(block.channels_mlp, w_block["channels_mlp"])
        _load_layernorm(block.norm2, w_block["norm2"])
        _load_mlp(block.tokens_mlp, w_block["tokens_mlp"])
    _load_layernorm(lane_enc.norm, w_lane["norm"])
    _load_mlp(lane_enc.emb_project, w_lane["emb_project"])
    
    # Fusion encoder
    fusion = encoder.fusion
    w_fusion = weights["fusion_encoder"]
    for i, block in enumerate(fusion.blocks):
        w_block = w_fusion["blocks"][i]
        _load_layernorm(block.norm1, w_block["norm1"])
        _load_multihead_attention(block.attn, w_block["attn"])
        _load_layernorm(block.norm2, w_block["norm2"])
        _load_mlp(block.mlp, w_block["mlp"])
    _load_layernorm(fusion.norm, w_fusion["norm"])
    
    # ========== Decoder ==========
    decoder = model.decoder.decoder
    dit = decoder.dit
    
    # Route encoder
    route_enc = dit.route_encoder
    w_route = weights["route_encoder"]
    _load_mlp(route_enc.channel_pre_project, w_route["channel_pre_project"])
    _load_mlp(route_enc.token_pre_project, w_route["token_pre_project"])
    _load_layernorm(route_enc.Mixer.norm1, w_route["mixer"]["norm1"])
    _load_mlp(route_enc.Mixer.channels_mlp, w_route["mixer"]["channels_mlp"])
    _load_layernorm(route_enc.Mixer.norm2, w_route["mixer"]["norm2"])
    _load_mlp(route_enc.Mixer.tokens_mlp, w_route["mixer"]["tokens_mlp"])
    _load_layernorm(route_enc.norm, w_route["norm"])
    _load_mlp(route_enc.emb_project, w_route["emb_project"])
    
    # Agent embedding
    _load_embedding(dit.agent_embedding, weights["agent_embedding"])
    
    # Preproj
    _load_mlp(dit.preproj, weights["preproj"])
    
    # Timestep embedder
    w_t = weights["t_embedder"]
    _load_linear(dit.t_embedder.mlp[0], w_t["mlp_0"])
    _load_linear(dit.t_embedder.mlp[2], w_t["mlp_2"])
    
    # DiT blocks
    for i, block in enumerate(dit.blocks):
        w_block = weights["dit_blocks"][i]
        _load_layernorm(block.norm1, w_block["norm1"])
        _load_multihead_attention(block.attn, w_block["attn"])
        _load_layernorm(block.norm2, w_block["norm2"])
        _load_mlp(block.mlp1, w_block["mlp1"])
        _load_linear(block.adaLN_modulation[1], w_block["adaLN_modulation"]["linear"])
        _load_layernorm(block.norm3, w_block["norm3"])
        _load_multihead_attention(block.cross_attn, w_block["cross_attn"])
        _load_layernorm(block.norm4, w_block["norm4"])
        _load_mlp(block.mlp2, w_block["mlp2"])
    
    # Final layer
    final = dit.final_layer
    w_final = weights["final_layer"]
    _load_layernorm(final.norm_final, w_final["norm_final"])
    _load_layernorm(final.proj[0], w_final["proj"]["norm1"])
    _load_linear(final.proj[1], w_final["proj"]["linear1"])
    _load_layernorm(final.proj[3], w_final["proj"]["norm2"])
    _load_linear(final.proj[4], w_final["proj"]["linear2"])
    _load_linear(final.adaLN_modulation[1], w_final["adaLN_modulation"]["linear"])


def main(skip_comparison: bool = False) -> None:
    """主函数"""
    json_path = os.environ.get("JSON_PATH", "dump_dit/dit.json")
    dump_dir = os.environ.get("DUMP_DIR", "dump_dit")
    os.makedirs(dump_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    
    # 加载 JSON
    print(f"Loading JSON from: {json_path}")
    j = _load_json(json_path)
    
    # 构建配置和模型
    cfg = _build_cfg(j["config"], device)
    model = Diffusion_Planner(cfg).to(device).eval()
    
    print(f"Config: hidden={cfg.hidden_dim}, depth={cfg.encoder_depth}, heads={cfg.num_heads}")
    
    # 加载输入
    inputs_j = j["inputs"]
    inputs = {
        'ego_current_state': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["ego_current_state"]).astype(np.float32)
        ).to(device),
        'neighbor_agents_past': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["neighbor_agents_past"]).astype(np.float32)
        ).to(device),
        'static_objects': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["static_objects"]).astype(np.float32)
        ).to(device),
        'lanes': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["lanes"]).astype(np.float32)
        ).to(device),
        'lanes_speed_limit': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["lanes_speed_limit"]).astype(np.float32)
        ).to(device),
        'lanes_has_speed_limit': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["lanes_has_speed_limit"]).astype(np.uint8)
        ).bool().to(device),
        'route_lanes': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["route_lanes"]).astype(np.float32)
        ).to(device),
        'diffusion_time': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["diffusion_time"]).astype(np.float32)
        ).to(device),
        'sampled_trajectories': torch.from_numpy(
            _b64_decode_to_numpy(inputs_j["sampled_trajectories"]).astype(np.float32)
        ).to(device),
    }
    
    B = inputs['ego_current_state'].shape[0]
    
    # 加载权重
    load_dit_weights(model, j["weights"])
    print("[OK] Weights loaded from JSON")
    
    # 预热
    warmup = int(os.environ.get("WARMUP", "1"))
    if warmup > 0:
        print(f"Running {warmup} warmup iterations...")
        model.train()
        with torch.no_grad():
            _ = model(inputs)
    
    # 计时前向传播
    print("Running timed forward pass...")
    model.train()  # 使用训练模式获取 score
    start_time = time.time()
    with torch.no_grad():
        encoder_outputs, decoder_outputs = model(inputs)
    elapsed = time.time() - start_time
    
    encoder_output = encoder_outputs['encoding'].cpu().numpy()
    score = decoder_outputs.get('score')
    if score is not None:
        score = score.cpu().numpy()
    
    # 打印结果
    print()
    print("=" * 60)
    print(f"[Forward] Encoder output shape: {encoder_output.shape}")
    if score is not None:
        print(f"[Forward] Decoder score shape: {score.shape}")
    print(f"[Timing] Forward pass: {elapsed * 1000:.2f}ms (warmup={warmup})")
    print(f"[Encoder] Min: {encoder_output.min():.6f}, Max: {encoder_output.max():.6f}, Mean: {encoder_output.mean():.6f}")
    if score is not None:
        print(f"[Score] Min: {score.min():.6f}, Max: {score.max():.6f}, Mean: {score.mean():.6f}")
    print("=" * 60)
    print()
    
    # 保存输出
    encoder_path = os.path.join(dump_dir, "encoder_output_torch.npy")
    np.save(encoder_path, encoder_output)
    print(f"[OK] Saved encoder output to: {encoder_path}")
    
    if score is not None:
        score_path = os.path.join(dump_dir, "decoder_score_torch.npy")
        np.save(score_path, score)
        print(f"[OK] Saved decoder score to: {score_path}")
    
    # 保存计时信息
    timing_info = {
        "platform": "pytorch",
        "elapsed_ms": elapsed * 1000,
        "batch_size": B,
        "hidden_dim": cfg.hidden_dim,
        "encoder_depth": cfg.encoder_depth,
        "decoder_depth": cfg.decoder_depth,
    }
    timing_path = os.path.join(dump_dir, "timing_torch.json")
    with open(timing_path, "w") as f:
        json.dump(timing_info, f, indent=2)
    print(f"[OK] Saved timing to: {timing_path}")


if __name__ == "__main__":
    main()

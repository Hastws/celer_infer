#!/usr/bin/env python3
"""
DiT (Diffusion Transformer) 模型权重导出器

将 PyTorch DiT 模型权重导出为 JSON 格式，供 C++ 推理引擎使用。

用法:
    python -m python.export.dit_dumper [--dump-dir dump_dit]
    
环境变量:
    DUMP_DIR: 输出目录 (默认: dump_dit)
    SEED: 随机种子 (默认: 42)
    B: 批次大小 (默认: 1)
    HIDDEN: hidden_dim (默认: 192)
    DEPTH: encoder/decoder depth (默认: 3)
    HEADS: num_heads (默认: 6)
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

from python.core.dit import (
    ModelConfig, Diffusion_Planner, create_dummy_inputs, count_parameters
)


def _b64_encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _encode_array(arr: np.ndarray, dtype: np.dtype, name: str = "") -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=dtype, order="C")
    raw = arr.tobytes(order="C")
    flat = arr.reshape(-1)
    preview_n = int(os.environ.get("JSON_PREVIEW_N", "8"))
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(np.dtype(dtype)),
        "encoding": "base64",
        "data": _b64_encode_bytes(raw),
        "preview": flat[: min(preview_n, flat.size)].tolist(),
    }


def _cfg_to_json_dict(cfg: ModelConfig) -> Dict[str, Any]:
    """将 ModelConfig 转换为 JSON 可序列化的字典"""
    return {
        "future_len": int(cfg.future_len),
        "time_len": int(cfg.time_len),
        "agent_state_dim": int(cfg.agent_state_dim),
        "agent_num": int(cfg.agent_num),
        "static_objects_state_dim": int(cfg.static_objects_state_dim),
        "static_objects_num": int(cfg.static_objects_num),
        "lane_len": int(cfg.lane_len),
        "lane_state_dim": int(cfg.lane_state_dim),
        "lane_num": int(cfg.lane_num),
        "route_len": int(cfg.route_len),
        "route_state_dim": int(cfg.route_state_dim),
        "route_num": int(cfg.route_num),
        "encoder_depth": int(cfg.encoder_depth),
        "decoder_depth": int(cfg.decoder_depth),
        "num_heads": int(cfg.num_heads),
        "hidden_dim": int(cfg.hidden_dim),
        "predicted_neighbor_num": int(cfg.predicted_neighbor_num),
        "diffusion_model_type": str(cfg.diffusion_model_type),
    }


def _fill_param(param: torch.Tensor, rng: np.random.RandomState, scale: float = 0.02) -> None:
    """用随机值填充参数"""
    arr = rng.standard_normal(size=tuple(param.shape)).astype(np.float32) * scale
    with torch.no_grad():
        param.copy_(torch.from_numpy(arr))


def _export_linear(linear: torch.nn.Linear, name: str) -> Dict[str, Any]:
    """导出 Linear 层"""
    result = {
        "weight": _encode_array(
            linear.weight.detach().cpu().numpy(), np.float32, f"{name}_weight"
        ),
    }
    if linear.bias is not None:
        result["bias"] = _encode_array(
            linear.bias.detach().cpu().numpy(), np.float32, f"{name}_bias"
        )
    return result


def _export_layernorm(ln: torch.nn.LayerNorm, name: str) -> Dict[str, Any]:
    """导出 LayerNorm 层"""
    return {
        "weight": _encode_array(
            ln.weight.detach().cpu().numpy(), np.float32, f"{name}_weight"
        ),
        "bias": _encode_array(
            ln.bias.detach().cpu().numpy(), np.float32, f"{name}_bias"
        ),
    }


def _export_embedding(emb: torch.nn.Embedding, name: str) -> Dict[str, Any]:
    """导出 Embedding 层"""
    return {
        "weight": _encode_array(
            emb.weight.detach().cpu().numpy(), np.float32, f"{name}_weight"
        ),
    }


def _export_mlp(mlp, name: str) -> Dict[str, Any]:
    """导出 timm Mlp 层 (fc1, act, fc2)"""
    return {
        "fc1": _export_linear(mlp.fc1, f"{name}_fc1"),
        "fc2": _export_linear(mlp.fc2, f"{name}_fc2"),
    }


def _export_multihead_attention(mha: torch.nn.MultiheadAttention, name: str) -> Dict[str, Any]:
    """导出 MultiheadAttention 层"""
    return {
        "in_proj_weight": _encode_array(
            mha.in_proj_weight.detach().cpu().numpy(), np.float32, f"{name}_in_proj_weight"
        ),
        "in_proj_bias": _encode_array(
            mha.in_proj_bias.detach().cpu().numpy(), np.float32, f"{name}_in_proj_bias"
        ),
        "out_proj": _export_linear(mha.out_proj, f"{name}_out_proj"),
    }


def export_dit_weights(model: Diffusion_Planner, cfg: ModelConfig) -> Dict[str, Any]:
    """导出完整的 DiT 模型权重"""
    weights = {}
    
    # ========== Encoder ==========
    encoder = model.encoder.encoder
    
    # Position embedding
    weights["encoder_pos_emb"] = _export_linear(encoder.pos_emb, "encoder_pos_emb")
    
    # Neighbor encoder (AgentFusionEncoder)
    neighbor_enc = encoder.neighbor_encoder
    weights["neighbor_encoder"] = {
        "type_emb": _export_linear(neighbor_enc.type_emb, "neighbor_type_emb"),
        "channel_pre_project": _export_mlp(neighbor_enc.channel_pre_project, "neighbor_channel_pre"),
        "token_pre_project": _export_mlp(neighbor_enc.token_pre_project, "neighbor_token_pre"),
        "blocks": [],
        "norm": _export_layernorm(neighbor_enc.norm, "neighbor_norm"),
        "emb_project": _export_mlp(neighbor_enc.emb_project, "neighbor_emb_proj"),
    }
    for i, block in enumerate(neighbor_enc.blocks):
        weights["neighbor_encoder"]["blocks"].append({
            "norm1": _export_layernorm(block.norm1, f"neighbor_block{i}_norm1"),
            "channels_mlp": _export_mlp(block.channels_mlp, f"neighbor_block{i}_channels_mlp"),
            "norm2": _export_layernorm(block.norm2, f"neighbor_block{i}_norm2"),
            "tokens_mlp": _export_mlp(block.tokens_mlp, f"neighbor_block{i}_tokens_mlp"),
        })
    
    # Static encoder (StaticFusionEncoder)
    static_enc = encoder.static_encoder
    weights["static_encoder"] = {
        "projection": _export_mlp(static_enc.projection, "static_projection"),
    }
    
    # Lane encoder (LaneFusionEncoder)
    lane_enc = encoder.lane_encoder
    weights["lane_encoder"] = {
        "speed_limit_emb": _export_linear(lane_enc.speed_limit_emb, "lane_speed_limit_emb"),
        "unknown_speed_emb": _export_embedding(lane_enc.unknown_speed_emb, "lane_unknown_speed_emb"),
        "traffic_emb": _export_linear(lane_enc.traffic_emb, "lane_traffic_emb"),
        "channel_pre_project": _export_mlp(lane_enc.channel_pre_project, "lane_channel_pre"),
        "token_pre_project": _export_mlp(lane_enc.token_pre_project, "lane_token_pre"),
        "blocks": [],
        "norm": _export_layernorm(lane_enc.norm, "lane_norm"),
        "emb_project": _export_mlp(lane_enc.emb_project, "lane_emb_proj"),
    }
    for i, block in enumerate(lane_enc.blocks):
        weights["lane_encoder"]["blocks"].append({
            "norm1": _export_layernorm(block.norm1, f"lane_block{i}_norm1"),
            "channels_mlp": _export_mlp(block.channels_mlp, f"lane_block{i}_channels_mlp"),
            "norm2": _export_layernorm(block.norm2, f"lane_block{i}_norm2"),
            "tokens_mlp": _export_mlp(block.tokens_mlp, f"lane_block{i}_tokens_mlp"),
        })
    
    # Fusion encoder (FusionEncoder with SelfAttentionBlocks)
    fusion = encoder.fusion
    weights["fusion_encoder"] = {
        "blocks": [],
        "norm": _export_layernorm(fusion.norm, "fusion_norm"),
    }
    for i, block in enumerate(fusion.blocks):
        weights["fusion_encoder"]["blocks"].append({
            "norm1": _export_layernorm(block.norm1, f"fusion_block{i}_norm1"),
            "attn": _export_multihead_attention(block.attn, f"fusion_block{i}_attn"),
            "norm2": _export_layernorm(block.norm2, f"fusion_block{i}_norm2"),
            "mlp": _export_mlp(block.mlp, f"fusion_block{i}_mlp"),
        })
    
    # ========== Decoder ==========
    decoder = model.decoder.decoder
    dit = decoder.dit
    
    # Route encoder
    route_enc = dit.route_encoder
    weights["route_encoder"] = {
        "channel_pre_project": _export_mlp(route_enc.channel_pre_project, "route_channel_pre"),
        "token_pre_project": _export_mlp(route_enc.token_pre_project, "route_token_pre"),
        "mixer": {
            "norm1": _export_layernorm(route_enc.Mixer.norm1, "route_mixer_norm1"),
            "channels_mlp": _export_mlp(route_enc.Mixer.channels_mlp, "route_mixer_channels_mlp"),
            "norm2": _export_layernorm(route_enc.Mixer.norm2, "route_mixer_norm2"),
            "tokens_mlp": _export_mlp(route_enc.Mixer.tokens_mlp, "route_mixer_tokens_mlp"),
        },
        "norm": _export_layernorm(route_enc.norm, "route_norm"),
        "emb_project": _export_mlp(route_enc.emb_project, "route_emb_proj"),
    }
    
    # Agent embedding
    weights["agent_embedding"] = _export_embedding(dit.agent_embedding, "agent_embedding")
    
    # Preproj
    weights["preproj"] = _export_mlp(dit.preproj, "preproj")
    
    # Timestep embedder
    weights["t_embedder"] = {
        "mlp_0": _export_linear(dit.t_embedder.mlp[0], "t_embedder_mlp_0"),
        "mlp_2": _export_linear(dit.t_embedder.mlp[2], "t_embedder_mlp_2"),
        "frequency_embedding_size": dit.t_embedder.frequency_embedding_size,
    }
    
    # DiT blocks
    weights["dit_blocks"] = []
    for i, block in enumerate(dit.blocks):
        weights["dit_blocks"].append({
            "norm1": _export_layernorm(block.norm1, f"dit_block{i}_norm1"),
            "attn": _export_multihead_attention(block.attn, f"dit_block{i}_attn"),
            "norm2": _export_layernorm(block.norm2, f"dit_block{i}_norm2"),
            "mlp1": _export_mlp(block.mlp1, f"dit_block{i}_mlp1"),
            "adaLN_modulation": {
                "linear": _export_linear(block.adaLN_modulation[1], f"dit_block{i}_adaLN"),
            },
            "norm3": _export_layernorm(block.norm3, f"dit_block{i}_norm3"),
            "cross_attn": _export_multihead_attention(block.cross_attn, f"dit_block{i}_cross_attn"),
            "norm4": _export_layernorm(block.norm4, f"dit_block{i}_norm4"),
            "mlp2": _export_mlp(block.mlp2, f"dit_block{i}_mlp2"),
        })
    
    # Final layer
    final = dit.final_layer
    weights["final_layer"] = {
        "norm_final": _export_layernorm(final.norm_final, "final_norm"),
        "proj": {
            "norm1": _export_layernorm(final.proj[0], "final_proj_norm1"),
            "linear1": _export_linear(final.proj[1], "final_proj_linear1"),
            "norm2": _export_layernorm(final.proj[3], "final_proj_norm2"),
            "linear2": _export_linear(final.proj[4], "final_proj_linear2"),
        },
        "adaLN_modulation": {
            "linear": _export_linear(final.adaLN_modulation[1], "final_adaLN"),
        },
    }
    
    return weights


class DiTDumper:
    """DiT 模型权重导出器类"""
    
    def __init__(self):
        pass
    
    def dump(self, model: Diffusion_Planner, cfg: ModelConfig, 
             inputs: Dict[str, torch.Tensor], output_dir: str = "dump_dit") -> str:
        """
        导出 DiT 模型权重到 JSON
        
        Args:
            model: Diffusion_Planner 模型实例
            cfg: 模型配置
            inputs: 输入张量字典
            output_dir: 输出目录
            
        Returns:
            生成的 JSON 文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        out_json = os.path.join(output_dir, "dit.json")
        
        # 获取参考输出
        model.eval()
        with torch.no_grad():
            encoder_outputs, decoder_outputs = model(inputs)
        
        # 构建 JSON
        j = {
            "meta": {
                "model_type": "dit",
                "seed": int(os.environ.get("SEED", "42")),
                "batch_size": inputs['ego_current_state'].shape[0],
            },
            "config": _cfg_to_json_dict(cfg),
            "inputs": {
                "ego_current_state": _encode_array(
                    inputs['ego_current_state'].cpu().numpy(), np.float32, "ego_current_state"
                ),
                "neighbor_agents_past": _encode_array(
                    inputs['neighbor_agents_past'].cpu().numpy(), np.float32, "neighbor_agents_past"
                ),
                "static_objects": _encode_array(
                    inputs['static_objects'].cpu().numpy(), np.float32, "static_objects"
                ),
                "lanes": _encode_array(
                    inputs['lanes'].cpu().numpy(), np.float32, "lanes"
                ),
                "lanes_speed_limit": _encode_array(
                    inputs['lanes_speed_limit'].cpu().numpy(), np.float32, "lanes_speed_limit"
                ),
                "lanes_has_speed_limit": _encode_array(
                    inputs['lanes_has_speed_limit'].cpu().numpy(), np.uint8, "lanes_has_speed_limit"
                ),
                "route_lanes": _encode_array(
                    inputs['route_lanes'].cpu().numpy(), np.float32, "route_lanes"
                ),
                "diffusion_time": _encode_array(
                    inputs['diffusion_time'].cpu().numpy(), np.float32, "diffusion_time"
                ),
                "sampled_trajectories": _encode_array(
                    inputs['sampled_trajectories'].cpu().numpy(), np.float32, "sampled_trajectories"
                ),
            },
            "weights": export_dit_weights(model, cfg),
            "reference": {
                "encoder_output": _encode_array(
                    encoder_outputs['encoding'].cpu().numpy(), np.float32, "encoder_output"
                ),
            },
        }
        
        # 根据训练/推理模式保存不同输出
        if 'score' in decoder_outputs:
            j["reference"]["decoder_score"] = _encode_array(
                decoder_outputs['score'].cpu().numpy(), np.float32, "decoder_score"
            )
        if 'prediction' in decoder_outputs:
            j["reference"]["decoder_prediction"] = _encode_array(
                decoder_outputs['prediction'].cpu().numpy(), np.float32, "decoder_prediction"
            )
        
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Exported DiT weights to: {out_json}")
        return out_json


def main() -> None:
    """主函数"""
    out_dir = os.environ.get("DUMP_DIR", "dump_dit")
    os.makedirs(out_dir, exist_ok=True)
    
    seed = int(os.environ.get("SEED", "42"))
    B = int(os.environ.get("B", "1"))
    
    # 模型配置
    hidden_dim = int(os.environ.get("HIDDEN", "192"))
    depth = int(os.environ.get("DEPTH", "3"))
    heads = int(os.environ.get("HEADS", "6"))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = ModelConfig(
        hidden_dim=hidden_dim,
        encoder_depth=depth,
        decoder_depth=depth,
        num_heads=heads,
        device=device,
    )
    
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Creating DiT model: hidden={hidden_dim}, depth={depth}, heads={heads}")
    model = Diffusion_Planner(cfg).to(device).eval()
    
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 创建输入
    inputs = create_dummy_inputs(cfg, batch_size=B, device=device)
    
    # 导出
    dumper = DiTDumper()
    json_path = dumper.dump(model, cfg, inputs, out_dir)
    
    # 保存 PyTorch 参考输出
    model.train()  # 切换到训练模式获取 score
    with torch.no_grad():
        encoder_outputs, decoder_outputs = model(inputs)
    
    encoder_output = encoder_outputs['encoding'].cpu().numpy()
    np.save(os.path.join(out_dir, "encoder_output_torch.npy"), encoder_output)
    
    if 'score' in decoder_outputs:
        score = decoder_outputs['score'].cpu().numpy()
        np.save(os.path.join(out_dir, "decoder_score_torch.npy"), score)
        print(f"[OK] Saved reference score shape: {score.shape}")
    
    print(f"[OK] Saved reference encoder output shape: {encoder_output.shape}")


if __name__ == "__main__":
    main()

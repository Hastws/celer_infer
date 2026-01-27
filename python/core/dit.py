"""
Standalone Diffusion Planner Model
可独立运行的模型文件，包含完整的模型定义和默认配置参数
"""

import math
import torch
import torch.nn as nn
from timm.layers import Mlp, DropPath


# ==================== Config ====================
class ModelConfig:
    """模型配置类，包含所有默认参数（与训练代码一致）"""
    def __init__(
        self,
        # Data dimensions
        future_len: int = 80,
        time_len: int = 21,
        agent_state_dim: int = 11,
        agent_num: int = 32,
        static_objects_state_dim: int = 10,
        static_objects_num: int = 5,
        lane_len: int = 20,
        lane_state_dim: int = 12,
        lane_num: int = 70,
        route_len: int = 20,
        route_state_dim: int = 12,
        route_num: int = 25,
        # Model architecture
        encoder_depth: int = 3,
        decoder_depth: int = 3,
        num_heads: int = 6,
        hidden_dim: int = 192,
        encoder_drop_path_rate: float = 0.1,
        decoder_drop_path_rate: float = 0.1,
        diffusion_model_type: str = 'x_start',
        # Decoder
        predicted_neighbor_num: int = 10,
        # Device
        device: str = 'cuda',
        # Normalizers (可选，推理时可设置为None)
        state_normalizer=None,
        observation_normalizer=None,
        guidance_fn=None,
    ):
        self.future_len = future_len
        self.time_len = time_len
        self.agent_state_dim = agent_state_dim
        self.agent_num = agent_num
        self.static_objects_state_dim = static_objects_state_dim
        self.static_objects_num = static_objects_num
        self.lane_len = lane_len
        self.lane_state_dim = lane_state_dim
        self.lane_num = lane_num
        self.route_len = route_len
        self.route_state_dim = route_state_dim
        self.route_num = route_num
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.encoder_drop_path_rate = encoder_drop_path_rate
        self.decoder_drop_path_rate = decoder_drop_path_rate
        self.diffusion_model_type = diffusion_model_type
        self.predicted_neighbor_num = predicted_neighbor_num
        self.device = device
        self.state_normalizer = state_normalizer
        self.observation_normalizer = observation_normalizer
        self.guidance_fn = guidance_fn


# ==================== Normalizers ====================
class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
    
    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    def __call__(self, data):
        from copy import copy
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data):
        from copy import copy
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data


# ==================== SDE ====================
class SDE:
    """SDE abstract class."""
    def __init__(self):
        super().__init__()

    @property
    def T(self):
        raise NotImplementedError

    def sde(self, x, t):
        raise NotImplementedError

    def marginal_prob(self, x, t):
        raise NotImplementedError

    def diffusion_coeff(self, t):
        raise NotImplementedError

    def marginal_prob_std(self, t):
        raise NotImplementedError


class VPSDE_linear(SDE):
    def __init__(self, beta_max=20.0, beta_min=0.1):
        super().__init__()
        self._beta_max = beta_max
        self._beta_min = beta_min

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        shape = x.shape
        reshape = [-1] + [1, ] * (len(shape) - 1)
        t = t.reshape(reshape)
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        drift = - 0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        shape = x.shape
        reshape = [-1] + [1, ] * (len(shape) - 1)
        t = t.reshape(reshape)
        mean_log_coeff = -0.25 * t ** 2 * (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t
        mean = torch.exp(mean_log_coeff) * x
        std = torch.sqrt(1 - torch.exp(2. * mean_log_coeff))
        return mean, std

    def diffusion_coeff(self, t):
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        diffusion = torch.sqrt(beta_t)
        return diffusion

    def marginal_prob_std(self, t):
        discount = torch.exp(-0.5 * t ** 2 * (self._beta_max - self._beta_min) - self._beta_min * t)
        std = torch.sqrt(1 - discount)
        return std


# ==================== Mixer Block ====================
class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    def forward(self, x):
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)


# ==================== DiT Components ====================
def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, cross_c, y, attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


# ==================== Encoder Components ====================
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):
    def __init__(self, time_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim
        self.type_emb = nn.Linear(3, channels_mlp_dim)
        self.channel_pre_project = Mlp(in_features=8+1, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]
        pos = x[:, :, -1, :7].clone()
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)
        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = torch.mean(x, dim=1)
        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)
        x = x + type_embedding
        x = self.emb_project(self.norm(x))
        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x
        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class StaticFusionEncoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        B, P, _ = x.shape
        pos = x[:, :, :7].clone()
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0
        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)
        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0
        valid_indices = ~mask_p.view(-1)
        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x
        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


class LaneFusionEncoder(nn.Module):
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()
        self._lane_len = lane_len
        self._channel = channels_mlp_dim
        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)
        self.channel_pre_project = Mlp(in_features=8, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]
        pos = x[:, :, int(self._lane_len / 2), :7].clone()
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)
        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = torch.mean(x, dim=1)
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)
        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit
        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand((~has_speed_limit).sum().item(), -1)
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)
        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))
        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x
        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device='cuda'):
        super().__init__()
        dpr = drop_path_rate
        self.blocks = nn.ModuleList([SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        mask[:, 0] = False
        for b in self.blocks:
            x = b(x, mask)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.token_num = config.agent_num + config.static_objects_num + config.lane_num
        self.neighbor_encoder = AgentFusionEncoder(config.time_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
        self.static_encoder = StaticFusionEncoder(config.static_objects_state_dim, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim)
        self.lane_encoder = LaneFusionEncoder(config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device
        )
        self.pos_emb = nn.Linear(7, config.hidden_dim)

    def forward(self, inputs):
        encoder_outputs = {}
        neighbors = inputs['neighbor_agents_past']
        static = inputs['static_objects']
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']
        B = neighbors.shape[0]
        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)
        encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim=1)
        encoding_pos = torch.cat([neighbor_pos, static_pos, lane_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)
        encoding_pos_result[~encoding_mask] = encoding_pos
        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)
        encoder_outputs['encoding'] = self.fusion(encoding_input, encoding_mask.view(B, self.token_num))
        return encoder_outputs


# ==================== Decoder Components ====================
class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()
        self._channel = channels_mlp_dim
        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        x = x[..., :4]
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)
        valid_indices = ~mask_b.view(-1)
        x = x[valid_indices]
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)
        x = torch.mean(x, dim=1)
        x = self.emb_project(self.norm(x))
        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x
        return x_result.view(B, -1)


class DiT(nn.Module):
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start"):
        super().__init__()
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        B, P, _ = x.shape
        x = self.preproj(x)
        x_embedding = torch.cat([self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)
        x = x + x_embedding
        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)
        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)
        x = self.final_layer(x, y)
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()
        self.dit = DiT(
            sde=self._sde,
            route_encoder=RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type
        )
        self._state_normalizer = config.state_normalizer
        self._observation_normalizer = config.observation_normalizer
        self._guidance_fn = config.guidance_fn

    @property
    def sde(self):
        return self._sde

    def forward(self, encoder_outputs, inputs):
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask
        current_states = torch.cat([ego_current, neighbors_current], dim=1)
        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)
        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']
        
        if self.training:
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1)
            diffusion_time = inputs['diffusion_time']
            return {
                "score": self.dit(
                    sampled_trajectories,
                    diffusion_time,
                    ego_neighbor_encoding,
                    route_lanes,
                    neighbor_current_mask
                ).reshape(B, P, -1, 4)
            }
        else:
            # 推理模式简化版本（不包含dpm_sampler）
            # 实际推理需要完整的dpm_sampler实现
            xT = torch.cat([current_states[:, :, None], torch.randn(B, P, self._future_len, 4).to(current_states.device) * 0.5], dim=2).reshape(B, P, -1)
            # 简化：直接返回一步预测（实际应使用迭代采样）
            x0 = self.dit(xT, torch.ones(B, device=xT.device) * 0.001, ego_neighbor_encoding, route_lanes, neighbor_current_mask)
            x0 = x0.reshape(B, P, -1, 4)[:, :, 1:]
            return {"prediction": x0}


# ==================== Main Model ====================
class Diffusion_Planner_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        return encoder_outputs


class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):
        decoder_outputs = self.decoder(encoder_outputs, inputs)
        return decoder_outputs


class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)

    @property
    def sde(self):
        return self.decoder.decoder.sde

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)
        return encoder_outputs, decoder_outputs


# ==================== Utility Functions ====================
def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dummy_inputs(config, batch_size=2, device='cuda'):
    """创建用于测试的虚拟输入数据"""
    B = batch_size
    inputs = {
        'ego_current_state': torch.randn(B, 10, device=device),
        'neighbor_agents_past': torch.randn(B, config.agent_num, config.time_len, config.agent_state_dim, device=device),
        'static_objects': torch.randn(B, config.static_objects_num, config.static_objects_state_dim, device=device),
        'lanes': torch.randn(B, config.lane_num, config.lane_len, config.lane_state_dim, device=device),
        'lanes_speed_limit': torch.randn(B, config.lane_num, 1, device=device),
        'lanes_has_speed_limit': torch.randint(0, 2, (B, config.lane_num, 1), device=device).bool(),
        'route_lanes': torch.randn(B, config.route_num, config.route_len, config.route_state_dim, device=device),
        'sampled_trajectories': torch.randn(B, 1 + config.predicted_neighbor_num, config.future_len + 1, 4, device=device),
        'diffusion_time': torch.rand(B, device=device),
    }
    return inputs


# ==================== Main Entry ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion Planner Standalone Model Test")
    print("=" * 60)
    
    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # 创建配置（使用默认参数，与训练代码一致）
    config = ModelConfig(device=device)
    
    print("\n--- Model Configuration ---")
    print(f"  future_len: {config.future_len}")
    print(f"  time_len: {config.time_len}")
    print(f"  agent_num: {config.agent_num}")
    print(f"  static_objects_num: {config.static_objects_num}")
    print(f"  lane_num: {config.lane_num}")
    print(f"  route_num: {config.route_num}")
    print(f"  encoder_depth: {config.encoder_depth}")
    print(f"  decoder_depth: {config.decoder_depth}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  predicted_neighbor_num: {config.predicted_neighbor_num}")
    print(f"  diffusion_model_type: {config.diffusion_model_type}")
    
    # 创建模型
    print("\n--- Creating Model ---")
    model = Diffusion_Planner(config).to(device)
    
    # 计算参数量
    total_params = count_parameters(model)
    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)
    
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Encoder parameters: {encoder_params:,} ({encoder_params/1e6:.2f}M)")
    print(f"  Decoder parameters: {decoder_params:,} ({decoder_params/1e6:.2f}M)")
    
    # 创建虚拟输入
    print("\n--- Testing Forward Pass (Training Mode) ---")
    batch_size = 2
    inputs = create_dummy_inputs(config, batch_size=batch_size, device=device)
    
    # 训练模式前向传播
    model.train()
    encoder_outputs, decoder_outputs = model(inputs)
    
    print(f"  Input batch size: {batch_size}")
    print(f"  Encoder output shape: {encoder_outputs['encoding'].shape}")
    print(f"  Decoder score shape: {decoder_outputs['score'].shape}")
    
    # 推理模式
    print("\n--- Testing Forward Pass (Inference Mode) ---")
    model.eval()
    with torch.no_grad():
        encoder_outputs, decoder_outputs = model(inputs)
    
    print(f"  Prediction shape: {decoder_outputs['prediction'].shape}")
    
    print("\n" + "=" * 60)
    print("Model test completed successfully!")
    print("=" * 60)

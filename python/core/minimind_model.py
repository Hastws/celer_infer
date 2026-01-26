import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


# =========================
# Config
# =========================
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = 0,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1_000_000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        # MoE
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if inference_rope_scaling
            else None
        )
        self.flash_attn = flash_attn

        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob


# =========================
# Utils: Norm / RoPE / KV repeat
# =========================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x_f).type_as(x)


def precompute_freqs_cis(
    dim: int, end: int, rope_base: float, rope_scaling: Optional[dict] = None
):
    # base freqs for half dim
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = float(rope_scaling.get("beta_fast", 32.0))
        beta_slow = float(rope_scaling.get("beta_slow", 1.0))
        attn_factor = float(rope_scaling.get("attention_factor", 1.0))

        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 1e-3),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    # q,k: (B,S,H,D) and (B,S,KVH,D)
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: (B,T,KVH,D) -> (B,T,H,D)
    if n_rep == 1:
        return x
    b, t, kvh, d = x.shape
    return x[:, :, :, None, :].expand(b, t, kvh, n_rep, d).reshape(b, t, kvh * n_rep, d)


# =========================
# Attention
# =========================
class Attention(nn.Module):
    def __init__(self, cfg: MiniMindConfig):
        super().__init__()
        kvh = (
            cfg.num_attention_heads
            if cfg.num_key_value_heads is None
            else cfg.num_key_value_heads
        )
        assert cfg.num_attention_heads % kvh == 0

        self.h = cfg.num_attention_heads
        self.kvh = kvh
        self.rep = self.h // self.kvh
        self.d = cfg.hidden_size // cfg.num_attention_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kvh * self.d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kvh * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, cfg.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout

        self.flash = hasattr(F, "scaled_dot_product_attention") and cfg.flash_attn

    def forward(
        self,
        x: torch.Tensor,  # (B,S,hidden)
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos,sin) each (S,D)
        past_key_value: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (k,v) each (B,T,KVH,D)
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,  # (B,T) 1=keep,0=mask
    ):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.h, self.d)
        k = self.k_proj(x).view(b, s, self.kvh, self.d)
        v = self.v_proj(x).view(b, s, self.kvh, self.d)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        # (B,S,H,D)->(B,H,S,D), k/v repeat to H then transpose
        q = q.transpose(1, 2)
        k = repeat_kv(k, self.rep).transpose(1, 2)
        v = repeat_kv(v, self.rep).transpose(1, 2)

        # flash path only for "clean" causal and no prefix cache
        if (
            self.flash
            and (s > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # scores: (B,H,S,T)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
            # causal on last s positions (same as your original)
            scores[:, :, :, -s:] += torch.triu(
                torch.full((s, s), float("-inf"), device=scores.device), diagonal=1
            )
            if attention_mask is not None:
                m = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                scores = scores + m

            probs = F.softmax(scores.float(), dim=-1).type_as(q)
            probs = self.attn_dropout(probs)
            out = probs @ v

        out = out.transpose(1, 2).reshape(b, s, -1)
        out = self.resid_dropout(self.o_proj(out))
        return out, past_kv


# =========================
# FFN / MoE
# =========================
class FeedForward(nn.Module):
    def __init__(self, cfg: MiniMindConfig):
        super().__init__()
        if cfg.intermediate_size is None:
            inter = int(cfg.hidden_size * 8 / 3)
            cfg.intermediate_size = 64 * ((inter + 63) // 64)

        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.act = ACT2FN[cfg.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        )


class MoEGate(nn.Module):
    def __init__(self, cfg: MiniMindConfig):
        super().__init__()
        self.cfg = cfg
        self.top_k = cfg.num_experts_per_tok
        self.E = cfg.n_routed_experts
        self.alpha = cfg.aux_loss_alpha
        self.seq_aux = cfg.seq_aux
        self.norm_topk_prob = cfg.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.E, cfg.hidden_size)))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        b, s, h = x.shape
        x2 = x.view(-1, h)
        logits = F.linear(x2, self.weight, None)
        if self.cfg.scoring_func != "softmax":
            raise NotImplementedError(
                f"Unsupported scoring_func: {self.cfg.scoring_func}"
            )
        scores = logits.softmax(dim=-1)

        topw, topi = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)

        # aux loss
        if self.training and self.alpha > 0.0:
            if self.seq_aux:
                topi_b = topi.view(b, -1)
                scores_b = scores.view(b, s, -1)
                ce = torch.zeros(b, self.E, device=x.device)
                ce.scatter_add_(
                    1, topi_b, torch.ones(b, s * self.top_k, device=x.device)
                )
                ce.div_(s * self.top_k / self.E)
                aux = (ce * scores_b.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                topi_b = topi.view(b, -1)
                ce = F.one_hot(topi_b.view(-1), num_classes=self.E).float().mean(0)
                Pi = scores.mean(0)
                aux = (Pi * (ce * self.E)).sum() * self.alpha
        else:
            aux = scores.new_zeros(())

        return topi, topw, aux


class MOEFeedForward(nn.Module):
    def __init__(self, cfg: MiniMindConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList(
            [FeedForward(cfg) for _ in range(cfg.n_routed_experts)]
        )
        self.gate = MoEGate(cfg)
        self.shared = (
            nn.ModuleList([FeedForward(cfg) for _ in range(cfg.n_shared_experts)])
            if cfg.n_shared_experts > 0
            else None
        )
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, s, h = x.shape
        topi, topw, aux = self.gate(x)
        self.aux_loss = aux

        x_flat = x.view(-1, h)
        flat_idx = topi.view(-1)

        if self.training:
            x_rep = x_flat.repeat_interleave(self.cfg.num_experts_per_tok, dim=0)
            y = torch.empty_like(x_rep)
            for i, exp in enumerate(self.experts):
                m = flat_idx == i
                out = exp(x_rep[m])
                if out.numel() > 0:
                    y[m] = out.to(y.dtype)
                else:
                    y[m] = out.to(y.dtype) + 0 * sum(p.sum() for p in exp.parameters())
            y = (y.view(*topw.shape, -1) * topw.unsqueeze(-1)).sum(dim=1).view(b, s, h)
        else:
            y = self._infer(x_flat, flat_idx, topw.view(-1, 1)).view(b, s, h)

        if self.shared is not None:
            for exp in self.shared:
                y = y + exp(identity)
        return y

    @torch.no_grad()
    def _infer(self, x_flat, flat_idx, flat_w):
        cache = torch.zeros_like(x_flat)
        order = flat_idx.argsort()
        ends = flat_idx.bincount().cpu().numpy().cumsum(0)
        token_idx = order // self.cfg.num_experts_per_tok

        for i, end in enumerate(ends):
            start = 0 if i == 0 else ends[i - 1]
            if start == end:
                continue
            exp = self.experts[i]
            t_idx = token_idx[start:end]
            out = exp(x_flat[t_idx]).to(cache.dtype)
            out.mul_(flat_w[order[start:end]])
            cache.scatter_add_(0, t_idx.view(-1, 1).repeat(1, x_flat.shape[-1]), out)
        return cache


# =========================
# Block / Model / LM
# =========================
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, cfg: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = MOEFeedForward(cfg) if cfg.use_moe else FeedForward(cfg)
        self.layer_id = layer_id

    def forward(self, x, pos_emb, past_kv=None, use_cache=False, attention_mask=None):
        r = x
        x, present = self.self_attn(
            self.input_layernorm(x), pos_emb, past_kv, use_cache, attention_mask
        )
        x = x + r
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class MiniMindModel(nn.Module):
    def __init__(self, cfg: MiniMindConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [MiniMindBlock(i, cfg) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        cos, sin = precompute_freqs_cis(
            dim=cfg.hidden_size // cfg.num_attention_heads,
            end=cfg.max_position_embeddings,
            rope_base=cfg.rope_theta,
            rope_scaling=cfg.rope_scaling,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(
        self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **_
    ):
        b, s = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        x = self.dropout(self.embed_tokens(input_ids))
        pos_emb = (
            self.freqs_cos[start_pos : start_pos + s],
            self.freqs_sin[start_pos : start_pos + s],
        )

        presents = []
        for layer, pkv in zip(self.layers, past_key_values):
            x, present = layer(x, pos_emb, pkv, use_cache, attention_mask)
            presents.append(present)

        x = self.norm(x)
        aux = sum(
            (l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)),
            x.new_zeros(()),
        )
        return x, presents, aux


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, cfg: Optional[MiniMindConfig] = None):
        cfg = cfg or MiniMindConfig()
        super().__init__(cfg)
        self.model = MiniMindModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # tie

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        hs, pkv, aux = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        sel = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hs[:, sel, :])

        loss = None
        if labels is not None:
            sl = logits[..., :-1, :].contiguous()
            lb = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100
            )

        out = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=pkv, hidden_states=hs
        )
        out.aux_loss = aux
        return out

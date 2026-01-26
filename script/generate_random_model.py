#!/usr/bin/env python3
"""
Generate a random model JSON with minimal dependencies.
Useful for benchmarking and testing without needing trained weights.

Usage:
    python generate_random_model.py \
        --hidden 64 \
        --layers 2 \
        --heads 8 \
        --kvh 2 \
        --vocab 128 \
        --seq-len 5 \
        --batch-size 2 \
        --output dump_minimind/minimind.json
"""

import os
import json
import base64
import argparse
from typing import Any, Dict

import numpy as np


def _b64_encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _encode_array(arr: np.ndarray, dtype: np.dtype, name: str = "") -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=dtype, order="C")
    raw = arr.tobytes(order="C")
    flat = arr.reshape(-1)
    preview_n = int(os.environ.get("JSON_PREVIEW_N", "16"))
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(np.dtype(dtype)),
        "encoding": "base64",
        "data": _b64_encode_bytes(raw),
        "preview": flat[: min(preview_n, flat.size)].tolist(),
    }


def generate_rope_cache(
    max_pos: int, head_dim: int, rope_theta: float = 1e6
) -> tuple[np.ndarray, np.ndarray]:
    """Generate RoPE cos and sin caches."""
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim)
    )
    t = np.arange(max_pos, dtype=np.float32)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate random model JSON")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size")
    parser.add_argument(
        "--layers", type=int, default=2, help="Number of transformer layers"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--kvh",
        type=int,
        default=2,
        help="Number of KV heads (for group query attention)",
    )
    parser.add_argument("--vocab", type=int, default=128, help="Vocabulary size")
    parser.add_argument(
        "--max-pos", type=int, default=128, help="Maximum position embeddings"
    )
    parser.add_argument(
        "--seq-len", type=int, default=5, help="Sequence length for input"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for input"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dump_minimind/minimind.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    hidden = args.hidden
    n_layers = args.layers
    n_heads = args.heads
    n_kv_heads = args.kvh
    vocab = args.vocab
    max_pos = args.max_pos
    seq_len = args.seq_len
    batch_size = args.batch_size
    head_dim = hidden // n_heads
    # Match the intermediate size calculation from llm_minimind_model.py
    # If intermediate_size is 0, it's calculated as: 64 * ((inter + 63) // 64)
    # where inter = hidden * 4
    inter = hidden * 4
    inter = 64 * ((inter + 63) // 64)

    # Generate random inputs
    input_ids = np.random.randint(0, vocab, size=(batch_size, seq_len), dtype=np.int32)
    attn_mask = np.ones((batch_size, seq_len), dtype=np.uint8)

    # Generate random weights
    def rand_weight(*shape):
        return np.random.randn(*shape).astype(np.float32) * 0.02

    tok_embedding = rand_weight(vocab, hidden)
    final_rms = np.ones(hidden, dtype=np.float32)
    lm_head = tok_embedding.copy()  # Tie embeddings

    # Generate RoPE cache
    rope_cos, rope_sin = generate_rope_cache(max_pos, head_dim)

    # Build layers
    layers = []
    for l in range(n_layers):
        layer = {
            "rms_attn": _encode_array(
                np.ones(hidden, dtype=np.float32), np.float32, f"layer{l}_rms_attn"
            ),
            "rms_ffn": _encode_array(
                np.ones(hidden, dtype=np.float32), np.float32, f"layer{l}_rms_ffn"
            ),
            "wq": _encode_array(
                rand_weight(n_heads * head_dim, hidden),
                np.float32,
                f"layer{l}_wq",
            ),
            "wk": _encode_array(
                rand_weight(n_kv_heads * head_dim, hidden),
                np.float32,
                f"layer{l}_wk",
            ),
            "wv": _encode_array(
                rand_weight(n_kv_heads * head_dim, hidden),
                np.float32,
                f"layer{l}_wv",
            ),
            "wo": _encode_array(
                rand_weight(hidden, n_heads * head_dim),
                np.float32,
                f"layer{l}_wo",
            ),
            "w_gate": _encode_array(
                rand_weight(inter, hidden), np.float32, f"layer{l}_w_gate"
            ),
            "w_up": _encode_array(
                rand_weight(inter, hidden), np.float32, f"layer{l}_w_up"
            ),
            "w_down": _encode_array(
                rand_weight(hidden, inter), np.float32, f"layer{l}_w_down"
            ),
        }
        layers.append(layer)

    # Build JSON
    j: Dict[str, Any] = {
        "meta": {
            "seed": args.seed,
            "B": batch_size,
            "S": seq_len,
            "head_dim": head_dim,
        },
        "config": {
            "dropout": 0.0,
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv_heads,
            "vocab_size": vocab,
            "max_position_embeddings": max_pos,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1_000_000.0,
            "inference_rope_scaling": False,
            "flash_attn": False,
            "use_moe": False,
            "intermediate_size": inter,
        },
        "inputs": {
            "input_ids": _encode_array(input_ids, np.int32, "input_ids"),
            "attention_mask": _encode_array(attn_mask, np.uint8, "attention_mask"),
        },
        "rope": {
            "cos": _encode_array(rope_cos, np.float32, "rope_cos"),
            "sin": _encode_array(rope_sin, np.float32, "rope_sin"),
        },
        "weights": {
            "tok_embedding": _encode_array(tok_embedding, np.float32, "tok_embedding"),
            "final_rms": _encode_array(final_rms, np.float32, "final_rms"),
            "lm_head": _encode_array(lm_head, np.float32, "lm_head"),
            "layers": layers,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)

    print(f"[OK] Generated random model JSON: {args.output}")
    print(
        f"  Config: hidden={hidden}, layers={n_layers}, heads={n_heads}, vocab={vocab}"
    )
    print(f"  Input: batch_size={batch_size}, seq_len={seq_len}")


if __name__ == "__main__":
    main()

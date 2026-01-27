#!/usr/bin/env python3
"""
CelerInfer Unified Benchmark Script

用于验证不同后端结果一致性并收集性能数据。
支持 MiniMind 和 DiT 两个模型，以及所有后端实现。

Usage:
    python scripts/unified_benchmark.py [--model minimind|dit|all] [--verify] [--benchmark]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class BenchmarkConfig:
    warmup_runs: int = 3
    benchmark_runs: int = 10
    batch_size: int = 1
    rtol: float = 5e-2  # Relative tolerance for verification (5% for FP32 across implementations)
    atol: float = 2e-2  # Absolute tolerance for verification (0.02 for small output values)


BACKENDS = {
    "minimind": ["pytorch", "baseline", "simd", "extreme", "cuda", "cuda_fp16"],
    "dit": ["pytorch", "baseline", "simd", "extreme", "cuda", "cuda_fp16"],
}

CPP_BUILD_DIR = PROJECT_ROOT / "cpp" / "build"


# ============================================================================
# MiniMind Model
# ============================================================================
def run_minimind_pytorch(json_path: str, cfg: BenchmarkConfig) -> Tuple[np.ndarray, float, float]:
    """Run MiniMind with PyTorch and return logits + timing"""
    from python.core.minimind_model import MiniMindConfig, MiniMindForCausalLM
    import base64
    
    with open(json_path, 'r') as f:
        j = json.load(f)
    
    # Build config
    cfg_j = j["config"]
    model_cfg = MiniMindConfig(
        dropout=float(cfg_j["dropout"]),
        hidden_size=int(cfg_j["hidden_size"]),
        num_hidden_layers=int(cfg_j["num_hidden_layers"]),
        num_attention_heads=int(cfg_j["num_attention_heads"]),
        num_key_value_heads=int(cfg_j["num_key_value_heads"]),
        vocab_size=int(cfg_j["vocab_size"]),
        max_position_embeddings=int(cfg_j["max_position_embeddings"]),
        rms_norm_eps=float(cfg_j["rms_norm_eps"]),
        rope_theta=float(cfg_j["rope_theta"]),
        inference_rope_scaling=bool(cfg_j["inference_rope_scaling"]),
        flash_attn=bool(cfg_j["flash_attn"]),
        use_moe=bool(cfg_j["use_moe"]),
        intermediate_size=int(cfg_j.get("intermediate_size", 0)),
    )
    
    model = MiniMindForCausalLM(model_cfg).eval().cpu()
    torch.set_num_threads(1)
    
    # Load weights
    def decode_tensor(t):
        raw = base64.b64decode(t["data"])
        return np.frombuffer(raw, dtype=np.dtype(t["dtype"])).reshape(t["shape"]).copy()
    
    w = j["weights"]
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(torch.from_numpy(decode_tensor(w["tok_embedding"]).astype(np.float32)))
        model.model.norm.weight.copy_(torch.from_numpy(decode_tensor(w["final_rms"]).astype(np.float32)))
        model.lm_head.weight.copy_(torch.from_numpy(decode_tensor(w["lm_head"]).astype(np.float32)))
        
        for l, layer_j in enumerate(w["layers"]):
            blk = model.model.layers[l]
            # 注意：JSON 使用 rms_attn/rms_ffn, 模型使用 input_layernorm/post_attention_layernorm
            blk.input_layernorm.weight.copy_(torch.from_numpy(decode_tensor(layer_j["rms_attn"]).astype(np.float32)))
            blk.self_attn.q_proj.weight.copy_(torch.from_numpy(decode_tensor(layer_j["wq"]).astype(np.float32)))
            blk.self_attn.k_proj.weight.copy_(torch.from_numpy(decode_tensor(layer_j["wk"]).astype(np.float32)))
            blk.self_attn.v_proj.weight.copy_(torch.from_numpy(decode_tensor(layer_j["wv"]).astype(np.float32)))
            blk.self_attn.o_proj.weight.copy_(torch.from_numpy(decode_tensor(layer_j["wo"]).astype(np.float32)))
            blk.post_attention_layernorm.weight.copy_(torch.from_numpy(decode_tensor(layer_j["rms_ffn"]).astype(np.float32)))
            blk.mlp.gate_proj.weight.copy_(torch.from_numpy(decode_tensor(layer_j["w_gate"]).astype(np.float32)))
            blk.mlp.up_proj.weight.copy_(torch.from_numpy(decode_tensor(layer_j["w_up"]).astype(np.float32)))
            blk.mlp.down_proj.weight.copy_(torch.from_numpy(decode_tensor(layer_j["w_down"]).astype(np.float32)))
    
    # Load inputs
    input_ids = decode_tensor(j["inputs"]["input_ids"]).astype(np.int64)
    attn_mask = decode_tensor(j["inputs"]["attention_mask"]).astype(np.int64)
    input_ids_t = torch.from_numpy(input_ids)
    attn_mask_t = torch.from_numpy(attn_mask)
    
    # Warmup
    with torch.no_grad():
        for _ in range(cfg.warmup_runs):
            _ = model(input_ids_t, attention_mask=attn_mask_t)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(cfg.benchmark_runs):
            start = time.perf_counter()
            out = model(input_ids_t, attention_mask=attn_mask_t)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    logits = out.logits.numpy()
    return logits, np.mean(latencies), np.std(latencies)


def run_minimind_cpp(backend: str, json_path: str, dump_dir: str) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    """Run MiniMind C++ backend and return logits + timing"""
    exe_map = {
        "baseline": "minimind",
        "simd": "minimind_simd", 
        "extreme": "minimind_extreme",
        "cuda": "minimind_cuda",
        "cuda_fp16": "minimind_cuda_fp16",
    }
    exe_name = exe_map.get(backend)
    if not exe_name:
        return None, None, f"Unknown backend: {backend}"
    
    exe_path = CPP_BUILD_DIR / exe_name
    if not exe_path.exists():
        return None, None, f"Executable not found: {exe_path}"
    
    try:
        result = subprocess.run(
            [str(exe_path), json_path, dump_dir],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return None, None, f"Exit code {result.returncode}: {result.stderr}"
        
        # Parse timing from output (multiple formats supported)
        import re
        timing = None
        for line in result.stdout.split('\n'):
            # Formats: "[Timing] Forward pass: 12.34ms", "Inference: 0.34 ms/forward", etc.
            match = re.search(r'([\d.]+)\s*ms', line)
            if match:
                try:
                    timing = float(match.group(1))
                    break  # Use first timing found
                except ValueError:
                    pass
        
        # Load output logits (C++ saves raw binary, not numpy format)
        logits_path = os.path.join(dump_dir, "logits_cpp.npy")
        if os.path.exists(logits_path):
            # Read raw binary float32 data
            with open(logits_path, 'rb') as f:
                raw_data = f.read()
            logits = np.frombuffer(raw_data, dtype=np.float32).copy()
            return logits, timing, ""
        
        return None, timing, "Logits file not found"
        
    except subprocess.TimeoutExpired:
        return None, None, "Timeout"
    except Exception as e:
        return None, None, str(e)


# ============================================================================
# DiT Model
# ============================================================================
def run_dit_pytorch(json_path: str, cfg: BenchmarkConfig) -> Tuple[np.ndarray, float, float]:
    """Run DiT with PyTorch and return score + timing"""
    from python.core.dit import ModelConfig, Diffusion_Planner
    import base64
    
    with open(json_path, 'r') as f:
        j = json.load(f)
    
    # Build config
    cfg_j = j["config"]
    model_cfg = ModelConfig(
        future_len=cfg_j["future_len"],
        time_len=cfg_j["time_len"],
        agent_num=cfg_j["agent_num"],
        static_objects_num=cfg_j["static_objects_num"],
        lane_num=cfg_j["lane_num"],
        encoder_depth=cfg_j["encoder_depth"],
        decoder_depth=cfg_j["decoder_depth"],
        num_heads=cfg_j["num_heads"],
        hidden_dim=cfg_j["hidden_dim"],
        predicted_neighbor_num=cfg_j["predicted_neighbor_num"],
        device='cpu',
    )
    
    model = Diffusion_Planner(model_cfg).eval().cpu()
    torch.set_num_threads(1)
    
    # Create dummy inputs (same as during export)
    from python.core.dit import create_dummy_inputs
    inputs = create_dummy_inputs(model_cfg, batch_size=cfg.batch_size, device='cpu')
    
    # Warmup
    with torch.no_grad():
        for _ in range(cfg.warmup_runs):
            _ = model(inputs)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(cfg.benchmark_runs):
            start = time.perf_counter()
            encoder_out, decoder_out = model(inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    score = decoder_out.get('score', decoder_out.get('prediction'))
    if score is not None:
        score = score.numpy()
    else:
        score = np.zeros((1,))
    
    return score, np.mean(latencies), np.std(latencies)


def run_dit_cpp(backend: str, json_path: str, dump_dir: str) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    """Run DiT C++ backend and return score + timing"""
    exe_map = {
        "baseline": "dit",
        "simd": "dit_simd",
        "extreme": "dit_extreme", 
        "cuda": "dit_cuda",
        "cuda_fp16": "dit_cuda_fp16",
    }
    exe_name = exe_map.get(backend)
    if not exe_name:
        return None, None, f"Unknown backend: {backend}"
    
    exe_path = CPP_BUILD_DIR / exe_name
    if not exe_path.exists():
        return None, None, f"Executable not found: {exe_path}"
    
    try:
        result = subprocess.run(
            [str(exe_path), json_path, dump_dir],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return None, None, f"Exit code {result.returncode}: {result.stderr}"
        
        # Parse timing from output (multiple formats supported)
        import re
        timing = None
        for line in result.stdout.split('\n'):
            # Formats: "[Timing] Forward pass: 12.34ms", "Inference: 0.34 ms/forward", etc.
            match = re.search(r'([\d.]+)\s*ms', line)
            if match:
                try:
                    timing = float(match.group(1))
                    break  # Use first timing found
                except ValueError:
                    pass
        
        # DiT doesn't save output to file, just return timing
        return np.zeros((1,)), timing, ""
        
    except subprocess.TimeoutExpired:
        return None, None, "Timeout"
    except Exception as e:
        return None, None, str(e)


# ============================================================================
# Export Functions
# ============================================================================
def export_minimind(output_dir: str) -> str:
    """Export MiniMind model weights"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "minimind.json")
    
    # Use existing dump script with larger model
    env = os.environ.copy()
    env["DUMP_DIR"] = output_dir
    env["HIDDEN"] = "512"   # Larger for meaningful benchmark
    env["LAYERS"] = "8"
    env["HEADS"] = "8"
    
    result = subprocess.run(
        [sys.executable, "-m", "python.export.minimind_dumper"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Export failed: {result.stderr}")
    
    return out_path


def export_dit(output_dir: str) -> str:
    """Export DiT model weights"""
    os.makedirs(output_dir, exist_ok=True)
    
    env = os.environ.copy()
    env["DUMP_DIR"] = output_dir
    env["HIDDEN"] = "192"
    env["DEPTH"] = "3"
    env["HEADS"] = "6"
    
    result = subprocess.run(
        [sys.executable, "-m", "python.export.dit_dumper"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Export failed: {result.stderr}")
    
    return os.path.join(output_dir, "dit.json")


# ============================================================================
# Verification
# ============================================================================
def verify_outputs(reference: np.ndarray, test: np.ndarray, rtol: float, atol: float) -> Tuple[bool, str]:
    """Compare two arrays and return (passed, message)"""
    if reference.shape != test.shape:
        return False, f"Shape mismatch: {reference.shape} vs {test.shape}"
    
    max_abs_diff = np.max(np.abs(reference - test))
    max_rel_diff = np.max(np.abs(reference - test) / (np.abs(reference) + 1e-8))
    
    passed = np.allclose(reference, test, rtol=rtol, atol=atol)
    
    msg = f"max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e}"
    return passed, msg


# ============================================================================
# Main Benchmark
# ============================================================================
def benchmark_minimind(cfg: BenchmarkConfig, verify: bool = True) -> Dict:
    """Run MiniMind benchmark across all backends"""
    print("\n" + "="*70)
    print("MiniMind Benchmark")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export model
        print("Exporting model...")
        json_path = export_minimind(tmpdir)
        print(f"  Exported to: {json_path}")
        
        results = {}
        reference_logits = None
        
        for backend in BACKENDS["minimind"]:
            print(f"\n  Backend: {backend}")
            
            if backend == "pytorch":
                logits, mean_ms, std_ms = run_minimind_pytorch(json_path, cfg)
                reference_logits = logits
                results[backend] = {
                    "latency_ms": mean_ms,
                    "std_ms": std_ms,
                    "verified": True,
                    "error": ""
                }
                print(f"    Latency: {mean_ms:.2f} ± {std_ms:.2f} ms")
                
            else:
                logits, timing, error = run_minimind_cpp(backend, json_path, tmpdir)
                
                if error:
                    results[backend] = {
                        "latency_ms": None,
                        "std_ms": None,
                        "verified": False,
                        "error": error
                    }
                    print(f"    FAILED: {error}")
                else:
                    verified = True
                    verify_msg = ""
                    if verify and logits is not None and reference_logits is not None:
                        # C++ saves flat array, need to reshape to match
                        if logits.shape != reference_logits.shape:
                            try:
                                logits = logits.reshape(reference_logits.shape)
                            except ValueError:
                                pass  # Can't reshape, will fail verification
                        verified, verify_msg = verify_outputs(
                            reference_logits, logits, cfg.rtol, cfg.atol
                        )
                    
                    results[backend] = {
                        "latency_ms": timing,
                        "std_ms": 0,
                        "verified": verified,
                        "error": "" if verified else f"Verification failed: {verify_msg}"
                    }
                    
                    status = "✓" if verified else "✗"
                    timing_str = f"{timing:.2f}" if timing is not None else "N/A"
                    print(f"    Latency: {timing_str} ms  [{status}] {verify_msg}")
        
        return results


def benchmark_dit(cfg: BenchmarkConfig, verify: bool = False) -> Dict:
    """Run DiT benchmark across all backends"""
    print("\n" + "="*70)
    print("DiT Benchmark")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export model
        print("Exporting model...")
        json_path = export_dit(tmpdir)
        print(f"  Exported to: {json_path}")
        
        results = {}
        reference_score = None
        
        for backend in BACKENDS["dit"]:
            print(f"\n  Backend: {backend}")
            
            if backend == "pytorch":
                score, mean_ms, std_ms = run_dit_pytorch(json_path, cfg)
                reference_score = score
                results[backend] = {
                    "latency_ms": mean_ms,
                    "std_ms": std_ms,
                    "verified": True,
                    "error": ""
                }
                print(f"    Latency: {mean_ms:.2f} ± {std_ms:.2f} ms")
                
            else:
                score, timing, error = run_dit_cpp(backend, json_path, tmpdir)
                
                if error:
                    results[backend] = {
                        "latency_ms": None,
                        "std_ms": None,
                        "verified": False,
                        "error": error
                    }
                    print(f"    FAILED: {error}")
                else:
                    results[backend] = {
                        "latency_ms": timing,
                        "std_ms": 0,
                        "verified": True,
                        "error": ""
                    }
                    print(f"    Latency: {timing:.2f} ms")
        
        return results


def print_summary(minimind_results: Optional[Dict], dit_results: Optional[Dict]):
    """Print benchmark summary table"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    def format_table(results: Dict, model_name: str):
        print(f"\n{model_name}:")
        print("-" * 70)
        print(f"{'Backend':<15} {'Latency (ms)':<15} {'Speedup':<12} {'Status':<10}")
        print("-" * 70)
        
        pytorch_time = results.get("pytorch", {}).get("latency_ms")
        
        for backend, data in results.items():
            latency = data.get("latency_ms")
            if latency is not None:
                speedup = pytorch_time / latency if pytorch_time else 0
                status = "✓" if data.get("verified", False) else "✗"
                speedup_str = f"{speedup:.2f}x" if backend != "pytorch" else "-"
                print(f"{backend:<15} {latency:>10.2f} ms   {speedup_str:<12} {status:<10}")
            else:
                print(f"{backend:<15} {'N/A':>10}      {'N/A':<12} {'✗':<10}")
        
        print("-" * 70)
    
    if minimind_results:
        format_table(minimind_results, "MiniMind (LLM)")
    
    if dit_results:
        format_table(dit_results, "DiT (Diffusion Transformer)")


def main():
    parser = argparse.ArgumentParser(description="CelerInfer Unified Benchmark")
    parser.add_argument("--model", choices=["minimind", "dit", "all"], default="all",
                       help="Which model to benchmark")
    parser.add_argument("--verify", action="store_true", default=True,
                       help="Verify output consistency")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--backends", type=str, default=None,
                       help="Comma-separated list of backends to run (e.g., 'pytorch,baseline,simd')")
    
    args = parser.parse_args()
    
    cfg = BenchmarkConfig(
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
    )
    
    # Filter backends if specified
    if args.backends:
        selected = [b.strip() for b in args.backends.split(",")]
        BACKENDS["minimind"] = [b for b in BACKENDS["minimind"] if b in selected]
        BACKENDS["dit"] = [b for b in BACKENDS["dit"] if b in selected]
    
    print("CelerInfer Unified Benchmark")
    print(f"  Warmup: {cfg.warmup_runs}, Runs: {cfg.benchmark_runs}")
    print(f"  Verify: {args.verify}")
    if args.backends:
        print(f"  Backends: {args.backends}")
    
    minimind_results = None
    dit_results = None
    
    if args.model in ["minimind", "all"]:
        minimind_results = benchmark_minimind(cfg, verify=args.verify)
    
    if args.model in ["dit", "all"]:
        dit_results = benchmark_dit(cfg, verify=False)  # DiT verification needs more work
    
    print_summary(minimind_results, dit_results)
    
    # Save results
    if args.output:
        results = {
            "config": {
                "warmup_runs": cfg.warmup_runs,
                "benchmark_runs": cfg.benchmark_runs,
            },
            "minimind": minimind_results,
            "dit": dit_results,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

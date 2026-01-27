#!/usr/bin/env python3
"""
Benchmark different sequence lengths across all backends.
Validates correctness by comparing outputs between backends.
"""

import subprocess
import sys
import time
import os
import json
import struct
import tempfile
import shutil
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'python'))

import torch
from core.minimind_model import MiniMindForCausalLM, MiniMindConfig

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class BenchmarkConfig:
    """Test configuration for small model"""
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    vocab_size: int = 128
    intermediate_size: int = 192
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0


# ============================================================================
# Weight Dumper (simplified inline version)
# ============================================================================
def dump_model_json(config: BenchmarkConfig, seq_len: int, output_path: str):
    """Dump model weights and config to JSON for C++ inference"""
    import base64
    
    # Create PyTorch config
    pt_config = MiniMindConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        vocab_size=config.vocab_size,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        flash_attn=False,
        use_moe=False,
    )
    
    # Create model with fixed seed for reproducibility
    torch.manual_seed(42)
    model = MiniMindForCausalLM(pt_config)
    model.eval()
    
    B, S = 1, seq_len
    input_ids = list(range(1, S + 1))
    input_ids = [x % config.vocab_size for x in input_ids]  # Ensure valid token ids
    
    def tensor_to_b64(t):
        arr = t.detach().cpu().numpy().astype(np.float32)
        return base64.b64encode(arr.tobytes()).decode('ascii')
    
    def int_tensor_to_b64(t):
        arr = t.detach().cpu().numpy().astype(np.int32)
        return base64.b64encode(arr.tobytes()).decode('ascii')
    
    def make_tensor_entry(t, name=""):
        arr = t.detach().cpu().numpy().astype(np.float32)
        return {
            "shape": list(arr.shape),
            "dtype": "float32",
            "data": base64.b64encode(arr.tobytes()).decode('ascii'),
            "preview": arr.flatten()[:5].tolist()
        }
    
    def uint8_tensor_to_b64(t):
        arr = t.detach().cpu().numpy().astype(np.uint8)
        return base64.b64encode(arr.tobytes()).decode('ascii')
    
    # Compute RoPE cache - use full max_position_embeddings
    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(config.max_position_embeddings).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cache = emb.cos()  # Full max_pos length
    sin_cache = emb.sin()  # Full max_pos length
    
    # Create attention mask (all 1s for no masking)
    attention_mask = torch.ones([B, S], dtype=torch.uint8)
    
    # Build JSON structure
    result = {
        "config": {
            "dropout": 0.0,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "inference_rope_scaling": False,
            "flash_attn": False,
            "use_moe": False,
            "intermediate_size": config.intermediate_size,
        },
        "meta": {"B": B, "S": S},
        "inputs": {
            "input_ids": {
                "shape": [B, S],
                "dtype": "int32",
                "data": int_tensor_to_b64(torch.tensor([input_ids], dtype=torch.int32)),
            },
            "attention_mask": {
                "shape": [B, S],
                "dtype": "uint8",
                "data": uint8_tensor_to_b64(attention_mask),
            }
        },
        "rope": {
            "cos": make_tensor_entry(cos_cache),
            "sin": make_tensor_entry(sin_cache),
        },
        "weights": {
            "tok_embedding": make_tensor_entry(model.model.embed_tokens.weight),
            "final_rms": make_tensor_entry(model.model.norm.weight),
            "lm_head": make_tensor_entry(model.lm_head.weight),
            "layers": []
        }
    }
    
    for i, layer in enumerate(model.model.layers):
        layer_data = {
            "rms_attn": make_tensor_entry(layer.input_layernorm.weight),
            "rms_ffn": make_tensor_entry(layer.post_attention_layernorm.weight),
            "wq": make_tensor_entry(layer.self_attn.q_proj.weight),
            "wk": make_tensor_entry(layer.self_attn.k_proj.weight),
            "wv": make_tensor_entry(layer.self_attn.v_proj.weight),
            "wo": make_tensor_entry(layer.self_attn.o_proj.weight),
            "w_gate": make_tensor_entry(layer.mlp.gate_proj.weight),
            "w_up": make_tensor_entry(layer.mlp.up_proj.weight),
            "w_down": make_tensor_entry(layer.mlp.down_proj.weight),
        }
        result["weights"]["layers"].append(layer_data)
    
    with open(output_path, 'w') as f:
        json.dump(result, f)
    
    # Return PyTorch reference output
    with torch.no_grad():
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        output = model(input_tensor)
        ref_logits = output.logits.numpy().flatten()
    
    return ref_logits, input_ids


# ============================================================================
# Benchmark Runner
# ============================================================================
def run_cpp_backend(executable: str, json_path: str, dump_dir: str, 
                    backend_flag: str = None, verbose: bool = False) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Run C++ backend and return timing + logits"""
    cmd = [executable, json_path, dump_dir]
    if backend_flag:
        cmd.append(backend_flag)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
        stderr = result.stderr
        
        if verbose or result.returncode != 0:
            print(f"\n  [DEBUG] Command: {' '.join(cmd)}")
            print(f"  [DEBUG] Return code: {result.returncode}")
            if stderr:
                print(f"  [DEBUG] stderr: {stderr[:200]}")
            if result.returncode != 0:
                print(f"  [DEBUG] stdout: {output[:500]}")
        
        # Parse timing
        time_ms = None
        for line in output.split('\n'):
            if '[Timing]' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'ms':
                        time_ms = float(parts[i-1])
                        break
        
        # Determine output file from stdout
        logits_file = None
        for line in output.split('\n'):
            if '[OK] Saved:' in line:
                logits_file = line.split('Saved:')[-1].strip()
                break
        
        # Fallback to guessing
        if logits_file is None or not os.path.exists(logits_file):
            if 'cuda' in executable.lower():
                logits_file = os.path.join(dump_dir, 'logits_cuda.npy')
            elif backend_flag:
                backend_name = backend_flag.replace('+', '_')
                logits_file = os.path.join(dump_dir, f'logits_{backend_name}.npy')
            else:
                logits_file = os.path.join(dump_dir, 'logits_cpp.npy')
        
        # Load logits
        logits = None
        if os.path.exists(logits_file):
            logits = np.fromfile(logits_file, dtype=np.float32)
        elif verbose:
            print(f"  [DEBUG] Logits file not found: {logits_file}")
        
        return time_ms, logits
    except Exception as e:
        print(f"Error running {executable}: {e}")
        return None, None


def run_pytorch_benchmark(config: BenchmarkConfig, input_ids: List[int], 
                          device: str, num_runs: int = 50) -> Tuple[float, np.ndarray]:
    """Run PyTorch inference"""
    pt_config = MiniMindConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        vocab_size=config.vocab_size,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        flash_attn=False,
        use_moe=False,
    )
    
    torch.manual_seed(42)
    model = MiniMindForCausalLM(pt_config).to(device)
    model.eval()
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    time_ms = (end - start) / num_runs * 1000
    
    logits = output.logits.detach().cpu().numpy().flatten()
    return time_ms, logits


def compare_logits(ref: np.ndarray, test: np.ndarray, name: str, 
                   tolerance: float = 1e-3) -> Tuple[bool, float]:
    """Compare logits and return (is_match, max_diff)"""
    if ref is None or test is None:
        return False, float('inf')
    
    if ref.shape != test.shape:
        # Try to match shapes
        min_len = min(len(ref), len(test))
        ref = ref[:min_len]
        test = test[:min_len]
    
    max_diff = np.abs(ref - test).max()
    is_match = max_diff < tolerance
    return is_match, max_diff


# ============================================================================
# Main Benchmark
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark across sequence lengths')
    parser.add_argument('--seq-lengths', '-s', type=int, nargs='+', 
                        default=[16, 32, 64, 128, 256],
                        help='Sequence lengths to test')
    parser.add_argument('--num-runs', '-n', type=int, default=100,
                        help='Number of runs per benchmark')
    parser.add_argument('--tolerance', '-t', type=float, default=0.02,
                        help='Max diff tolerance for correctness')
    args = parser.parse_args()
    
    config = BenchmarkConfig()
    cpp_build = os.path.join(project_root, 'cpp', 'build')
    
    # Check available executables
    executables = {
        'unified': os.path.join(cpp_build, 'minimind_unified'),
        'cuda': os.path.join(cpp_build, 'minimind_cuda'),
    }
    
    available = {k: v for k, v in executables.items() if os.path.exists(v)}
    
    print("=" * 80)
    print("CelerInfer Multi-Sequence-Length Benchmark")
    print("=" * 80)
    print(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, vocab={config.vocab_size}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Runs per test: {args.num_runs}")
    print(f"Tolerance for correctness: {args.tolerance}")
    print()
    
    # Results storage
    all_results = []
    
    for seq_len in args.seq_lengths:
        print(f"\n{'='*80}")
        print(f"Testing Sequence Length: {seq_len}")
        print('='*80)
        
        # Create temp directory for this test
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, 'model.json')
            
            # Dump model
            print(f"  Dumping model weights...", end=" ", flush=True)
            ref_logits, input_ids = dump_model_json(config, seq_len, json_path)
            print(f"✓ (ref shape: {ref_logits.shape})")
            
            results = {'seq_len': seq_len}
            
            # PyTorch CPU
            print(f"  PyTorch CPU...", end=" ", flush=True)
            try:
                pt_cpu_time, pt_cpu_logits = run_pytorch_benchmark(
                    config, input_ids, 'cpu', args.num_runs)
                results['pytorch_cpu'] = pt_cpu_time
                match, diff = compare_logits(ref_logits, pt_cpu_logits, "PyTorch CPU", args.tolerance)
                results['pytorch_cpu_match'] = match
                results['pytorch_cpu_diff'] = diff
                print(f"✓ {pt_cpu_time:.3f} ms (diff={diff:.2e})")
            except Exception as e:
                print(f"✗ {e}")
                results['pytorch_cpu'] = None
            
            # PyTorch CUDA
            print(f"  PyTorch CUDA...", end=" ", flush=True)
            if torch.cuda.is_available():
                try:
                    pt_cuda_time, pt_cuda_logits = run_pytorch_benchmark(
                        config, input_ids, 'cuda', args.num_runs)
                    results['pytorch_cuda'] = pt_cuda_time
                    match, diff = compare_logits(ref_logits, pt_cuda_logits, "PyTorch CUDA", args.tolerance)
                    results['pytorch_cuda_match'] = match
                    results['pytorch_cuda_diff'] = diff
                    print(f"✓ {pt_cuda_time:.3f} ms (diff={diff:.2e})")
                except Exception as e:
                    print(f"✗ {e}")
                    results['pytorch_cuda'] = None
            else:
                print("✗ Not available")
                results['pytorch_cuda'] = None
            
            # C++ unified backends
            if 'unified' in available:
                for backend in ['baseline', 'simd', 'quant', 'simd+quant']:
                    print(f"  C++ {backend}...", end=" ", flush=True)
                    time_ms, logits = run_cpp_backend(
                        available['unified'], json_path, tmp_dir, backend, verbose=False)
                    key = f'cpp_{backend.replace("+", "_")}'
                    results[key] = time_ms
                    if logits is not None:
                        match, diff = compare_logits(ref_logits, logits, f"C++ {backend}", args.tolerance)
                        results[f'{key}_match'] = match
                        results[f'{key}_diff'] = diff
                        status = "✓" if match else "⚠"
                        print(f"{status} {time_ms:.3f} ms (diff={diff:.2e})")
                    else:
                        print(f"✗ Failed")
            
            # C++ CUDA
            if 'cuda' in available:
                print(f"  C++ CUDA...", end=" ", flush=True)
                time_ms, logits = run_cpp_backend(
                    available['cuda'], json_path, tmp_dir)
                results['cpp_cuda'] = time_ms
                if logits is not None:
                    match, diff = compare_logits(ref_logits, logits, "C++ CUDA", args.tolerance)
                    results['cpp_cuda_match'] = match
                    results['cpp_cuda_diff'] = diff
                    status = "✓" if match else "⚠"
                    print(f"{status} {time_ms:.3f} ms (diff={diff:.2e})")
                else:
                    print(f"✗ Failed")
            
            all_results.append(results)
    
    # ========================================================================
    # Generate Report
    # ========================================================================
    print("\n")
    print("=" * 100)
    print("BENCHMARK REPORT")
    print("=" * 100)
    
    # Timing table
    print("\n📊 TIMING RESULTS (ms)")
    print("-" * 100)
    header = f"{'Seq Len':<10}"
    backends = ['pytorch_cpu', 'pytorch_cuda', 'cpp_baseline', 'cpp_simd', 
                'cpp_quant', 'cpp_simd_quant', 'cpp_cuda']
    backend_names = ['PT CPU', 'PT CUDA', 'C++ Base', 'C++ SIMD', 
                     'C++ Quant', 'C++ S+Q', 'C++ CUDA']
    
    for name in backend_names:
        header += f"{name:>12}"
    print(header)
    print("-" * 100)
    
    for res in all_results:
        row = f"{res['seq_len']:<10}"
        for backend in backends:
            val = res.get(backend)
            if val is not None:
                row += f"{val:>12.3f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    # Speedup table
    print("\n🚀 SPEEDUP vs PyTorch CPU")
    print("-" * 100)
    header = f"{'Seq Len':<10}"
    for name in backend_names:
        header += f"{name:>12}"
    print(header)
    print("-" * 100)
    
    for res in all_results:
        row = f"{res['seq_len']:<10}"
        baseline = res.get('pytorch_cpu')
        for backend in backends:
            val = res.get(backend)
            if val is not None and baseline is not None:
                speedup = baseline / val
                row += f"{speedup:>11.1f}x"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    # Correctness table
    print("\n✅ CORRECTNESS CHECK (max diff vs PyTorch reference)")
    print("-" * 100)
    header = f"{'Seq Len':<10}"
    for name in backend_names:
        header += f"{name:>12}"
    print(header)
    print("-" * 100)
    
    for res in all_results:
        row = f"{res['seq_len']:<10}"
        for backend in backends:
            diff_key = f'{backend}_diff'
            match_key = f'{backend}_match'
            diff = res.get(diff_key)
            match = res.get(match_key)
            if diff is not None:
                symbol = "✓" if match else "⚠"
                row += f"{symbol}{diff:>10.2e}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    # Summary
    print("\n" + "=" * 100)
    print("📈 SUMMARY")
    print("=" * 100)
    
    # Find fastest backend per seq length
    print("\n🏆 Fastest Backend per Sequence Length:")
    for res in all_results:
        times = [(k, v) for k, v in res.items() 
                 if k in backends and v is not None]
        if times:
            fastest = min(times, key=lambda x: x[1])
            print(f"  S={res['seq_len']:>3}: {fastest[0]:<15} @ {fastest[1]:.3f} ms")
    
    # CUDA vs SIMD comparison
    print("\n⚡ CUDA vs SIMD Performance:")
    for res in all_results:
        cuda = res.get('cpp_cuda')
        simd = res.get('cpp_simd')
        if cuda and simd:
            ratio = simd / cuda
            print(f"  S={res['seq_len']:>3}: CUDA is {ratio:.2f}x faster than SIMD "
                  f"({cuda:.3f} ms vs {simd:.3f} ms)")
    
    # All results match check
    print("\n🔍 Correctness Summary:")
    all_match = True
    for res in all_results:
        for backend in backends:
            match = res.get(f'{backend}_match')
            if match is False:
                all_match = False
                diff = res.get(f'{backend}_diff', 'N/A')
                print(f"  ⚠ S={res['seq_len']}, {backend}: diff={diff:.2e} exceeds tolerance")
    
    if all_match:
        print("  ✓ All backends produce consistent results within tolerance!")
    
    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()

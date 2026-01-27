#!/usr/bin/env python3
"""
Comprehensive benchmark comparing:
- PyTorch CPU
- PyTorch CUDA  
- C++ baseline
- C++ SIMD
- C++ unified (all combinations)
- C++ CUDA (cuBLAS)
"""

import subprocess
import sys
import time
import os
import argparse
import struct
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'python'))

from core.minimind_model import MiniMindForCausalLM, MiniMindConfig

import torch

def run_pytorch_benchmark(config, input_ids, device, num_runs=100, warmup=5):
    """Run PyTorch inference and measure time"""
    model = MiniMindForCausalLM(config).to(device)
    model.eval()
    
    # Move input to device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Synchronize if CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(input_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    elapsed_ms = (end - start) / num_runs * 1000
    
    # Get logits stats - outputs is CausalLMOutputWithPast, need .logits
    logits_np = outputs.logits.detach().cpu().numpy()
    
    return {
        'time_ms': elapsed_ms,
        'logits': logits_np,
        'min': float(logits_np.min()),
        'max': float(logits_np.max()),
        'mean': float(logits_np.mean())
    }

def run_cpp_benchmark(executable_path, json_path, dump_dir, backend_flag=None):
    """Run C++ executable and parse output"""
    cmd = [executable_path, json_path, dump_dir]
    if backend_flag:
        cmd.append(backend_flag)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Parse timing from output
        for line in output.split('\n'):
            if '[Timing]' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'ms':
                        return {
                            'time_ms': float(parts[i-1]),
                            'output': output,
                            'success': True
                        }
        
        return {'time_ms': None, 'output': output, 'success': False}
    except Exception as e:
        return {'time_ms': None, 'output': str(e), 'success': False}

def load_cpp_logits(filepath):
    """Load logits saved by C++ inference"""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        logits = np.frombuffer(data, dtype=np.float32)
        return logits
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description='Full benchmark suite')
    parser.add_argument('--batch-size', '-B', type=int, default=1, help='Batch size')
    parser.add_argument('--seq-len', '-S', type=int, default=32, help='Sequence length')
    parser.add_argument('--num-runs', '-n', type=int, default=100, help='Number of runs')
    parser.add_argument('--json-path', type=str, default='dump_minimind/minimind.json')
    parser.add_argument('--dump-dir', type=str, default='dump_minimind')
    args = parser.parse_args()
    
    print("="*70)
    print("CelerInfer Full Benchmark Suite")
    print("="*70)
    print(f"Config: B={args.batch_size}, S={args.seq_len}, runs={args.num_runs}")
    print()
    
    # Model config
    config = MiniMindConfig()
    input_ids = list(range(1, args.seq_len + 1))
    
    results = {}
    
    # ============================================================
    # PyTorch CPU
    # ============================================================
    print("[1/7] PyTorch CPU...", end=" ", flush=True)
    try:
        result = run_pytorch_benchmark(config, input_ids, 'cpu', args.num_runs)
        results['pytorch_cpu'] = result
        print(f"✓ {result['time_ms']:.3f} ms")
    except Exception as e:
        print(f"✗ {e}")
        results['pytorch_cpu'] = None
    
    # ============================================================
    # PyTorch CUDA
    # ============================================================
    print("[2/7] PyTorch CUDA...", end=" ", flush=True)
    if torch.cuda.is_available():
        try:
            result = run_pytorch_benchmark(config, input_ids, 'cuda', args.num_runs)
            results['pytorch_cuda'] = result
            print(f"✓ {result['time_ms']:.3f} ms")
        except Exception as e:
            print(f"✗ {e}")
            results['pytorch_cuda'] = None
    else:
        print("✗ CUDA not available")
        results['pytorch_cuda'] = None
    
    # C++ executables
    cpp_build = os.path.join(project_root, 'cpp', 'build')
    
    # ============================================================
    # C++ baseline
    # ============================================================
    print("[3/7] C++ baseline...", end=" ", flush=True)
    exe_path = os.path.join(cpp_build, 'minimind')
    if os.path.exists(exe_path):
        result = run_cpp_benchmark(exe_path, args.json_path, args.dump_dir)
        results['cpp_baseline'] = result
        if result['success']:
            print(f"✓ {result['time_ms']:.3f} ms")
        else:
            print(f"✗ Failed")
    else:
        print("✗ Not built")
        results['cpp_baseline'] = None
    
    # ============================================================
    # C++ SIMD
    # ============================================================
    print("[4/7] C++ SIMD...", end=" ", flush=True)
    exe_path = os.path.join(cpp_build, 'minimind_simd')
    if os.path.exists(exe_path):
        result = run_cpp_benchmark(exe_path, args.json_path, args.dump_dir)
        results['cpp_simd'] = result
        if result['success']:
            print(f"✓ {result['time_ms']:.3f} ms")
        else:
            print(f"✗ Failed")
    else:
        print("✗ Not built")
        results['cpp_simd'] = None
    
    # ============================================================
    # C++ unified with different backends
    # ============================================================
    unified_backends = [
        ('baseline', 'cpp_unified_baseline'),
        ('simd', 'cpp_unified_simd'),
        ('quant', 'cpp_unified_quant'),
        ('simd+quant', 'cpp_unified_simd_quant'),
    ]
    
    exe_path = os.path.join(cpp_build, 'minimind_unified')
    if os.path.exists(exe_path):
        for backend_flag, name in unified_backends:
            print(f"[5] C++ unified ({backend_flag})...", end=" ", flush=True)
            result = run_cpp_benchmark(exe_path, args.json_path, args.dump_dir, backend_flag)
            results[name] = result
            if result['success']:
                print(f"✓ {result['time_ms']:.3f} ms")
            else:
                print(f"✗ Failed")
    else:
        for _, name in unified_backends:
            results[name] = None
    
    # ============================================================
    # C++ CUDA
    # ============================================================
    print("[6/7] C++ CUDA...", end=" ", flush=True)
    exe_path = os.path.join(cpp_build, 'minimind_cuda')
    if os.path.exists(exe_path):
        result = run_cpp_benchmark(exe_path, args.json_path, args.dump_dir)
        results['cpp_cuda'] = result
        if result['success']:
            print(f"✓ {result['time_ms']:.3f} ms")
        else:
            print(f"✗ Failed")
    else:
        print("✗ Not built")
        results['cpp_cuda'] = None
    
    # ============================================================
    # Summary
    # ============================================================
    print()
    print("="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Backend':<25} {'Time (ms)':<12} {'Speedup':<10} {'vs PyTorch CPU':<15}")
    print("-"*70)
    
    # Get baseline for comparison
    baseline_time = results.get('pytorch_cpu', {}).get('time_ms') if results.get('pytorch_cpu') else None
    simd_time = results.get('cpp_unified_simd', {}).get('time_ms') if results.get('cpp_unified_simd') else None
    cuda_time = results.get('cpp_cuda', {}).get('time_ms') if results.get('cpp_cuda') else None
    
    def format_row(name, result, baseline_time):
        if result is None or result.get('time_ms') is None:
            return f"{name:<25} {'N/A':<12} {'-':<10} {'-':<15}"
        
        time_ms = result['time_ms']
        if baseline_time:
            speedup = baseline_time / time_ms
            return f"{name:<25} {time_ms:<12.3f} {speedup:<10.2f}x {'':<15}"
        else:
            return f"{name:<25} {time_ms:<12.3f} {'-':<10} {'':<15}"
    
    # Print all results
    print(format_row("PyTorch CPU", results.get('pytorch_cpu'), baseline_time))
    print(format_row("PyTorch CUDA", results.get('pytorch_cuda'), baseline_time))
    print("-"*70)
    print(format_row("C++ baseline", results.get('cpp_baseline'), baseline_time))
    print(format_row("C++ SIMD (AVX2)", results.get('cpp_simd'), baseline_time))
    print(format_row("C++ unified baseline", results.get('cpp_unified_baseline'), baseline_time))
    print(format_row("C++ unified SIMD", results.get('cpp_unified_simd'), baseline_time))
    print(format_row("C++ unified quant", results.get('cpp_unified_quant'), baseline_time))
    print(format_row("C++ unified SIMD+quant", results.get('cpp_unified_simd_quant'), baseline_time))
    print("-"*70)
    print(format_row("C++ CUDA (cuBLAS)", results.get('cpp_cuda'), baseline_time))
    print("="*70)
    
    # CUDA vs SIMD comparison
    if cuda_time and simd_time:
        print()
        print("🔥 CUDA vs SIMD comparison:")
        print(f"   C++ SIMD:  {simd_time:.3f} ms")
        print(f"   C++ CUDA:  {cuda_time:.3f} ms")
        print(f"   CUDA is {simd_time/cuda_time:.2f}x faster than SIMD")
    
    pytorch_cuda_time = results.get('pytorch_cuda', {}).get('time_ms') if results.get('pytorch_cuda') else None
    if cuda_time and pytorch_cuda_time:
        print()
        print("🔥 C++ CUDA vs PyTorch CUDA comparison:")
        print(f"   PyTorch CUDA:  {pytorch_cuda_time:.3f} ms")
        print(f"   C++ CUDA:      {cuda_time:.3f} ms")
        if pytorch_cuda_time > cuda_time:
            print(f"   C++ CUDA is {pytorch_cuda_time/cuda_time:.2f}x faster than PyTorch CUDA")
        else:
            print(f"   PyTorch CUDA is {cuda_time/pytorch_cuda_time:.2f}x faster than C++ CUDA")
    
    # CUDA vs SIMD
    if cuda_time and simd_time:
        print()
        print("🚀 C++ CUDA vs C++ SIMD comparison:")
        print(f"   C++ SIMD:  {simd_time:.3f} ms")
        print(f"   C++ CUDA:  {cuda_time:.3f} ms")
        if simd_time > cuda_time:
            print(f"   CUDA is {simd_time/cuda_time:.2f}x faster than SIMD")
        else:
            print(f"   SIMD is {cuda_time/simd_time:.2f}x faster than CUDA")
    
    # Summary
    print()
    print("="*70)
    print("📊 Performance Summary (vs PyTorch CPU baseline):")
    print("="*70)
    print(f"   PyTorch CPU:       {baseline_time:.3f} ms (1.0x)")
    if pytorch_cuda_time:
        print(f"   PyTorch CUDA:      {pytorch_cuda_time:.3f} ms ({baseline_time/pytorch_cuda_time:.1f}x)")
    
    cpp_unified_simd = results.get('cpp_unified_simd', {}).get('time_ms') if results.get('cpp_unified_simd') else None
    if cpp_unified_simd:
        print(f"   C++ SIMD (AVX2):   {cpp_unified_simd:.3f} ms ({baseline_time/cpp_unified_simd:.1f}x)")
    if cuda_time:
        print(f"   C++ CUDA (cuBLAS): {cuda_time:.3f} ms ({baseline_time/cuda_time:.1f}x) 🏆 FASTEST")
    print("="*70)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Multi-Backend Benchmark Tool for CelerInfer

This tool benchmarks all backend combinations of the unified inference engine:
  - baseline: Pure C++ implementation
  - simd: AVX2 vectorized operations
  - fusion: Operator fusion
  - quant: INT8 quantization  
  - All combinations: simd+fusion, simd+quant, fusion+quant, etc.

Usage:
    python -m python.tools.benchmark_backends [--backends all] [--test-case basic]
    
Examples:
    python -m python.tools.benchmark_backends --backends baseline,simd,simd+fusion
    python -m python.tools.benchmark_backends --backends all --test-case batch8
"""

import os
import sys
import re
import json
import struct
import subprocess
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class TestCase:
    """Test case configuration"""
    name: str
    B: int
    S: int
    seed: int = 123
    description: str = ""


@dataclass 
class BackendResult:
    """Result from a single backend run"""
    backend: str
    elapsed_ms: float
    max_logit: float
    min_logit: float
    mean_logit: float
    success: bool
    error: str = ""


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a test case"""
    test_case: TestCase
    pytorch_ms: float
    pytorch_logits_path: str
    backend_results: List[BackendResult] = field(default_factory=list)
    
    def get_speedups(self) -> Dict[str, float]:
        """Calculate speedup vs PyTorch for each backend"""
        speedups = {}
        for br in self.backend_results:
            if br.success and br.elapsed_ms > 0 and self.pytorch_ms > 0:
                speedups[br.backend] = self.pytorch_ms / br.elapsed_ms
        return speedups
    
    def get_vs_baseline(self) -> Dict[str, float]:
        """Calculate speedup vs baseline for each backend"""
        baseline_ms = None
        for br in self.backend_results:
            if br.backend == "baseline" and br.success:
                baseline_ms = br.elapsed_ms
                break
        
        if baseline_ms is None or baseline_ms <= 0:
            return {}
            
        speedups = {}
        for br in self.backend_results:
            if br.success and br.elapsed_ms > 0:
                speedups[br.backend] = baseline_ms / br.elapsed_ms
        return speedups


# Available backends  
ALL_BACKENDS = [
    "baseline",
    "simd",
    "fusion", 
    "quant",
    "simd+fusion",
    "simd+quant",
    "fusion+quant",
    "simd+fusion+quant",
]

# Test cases
TEST_CASES = {
    "basic": TestCase(name="basic", B=2, S=5, seed=123, description="Default test (B=2, S=5)"),
    "batch1": TestCase(name="batch1", B=1, S=8, seed=456, description="Single batch (B=1, S=8)"),
    "batch4": TestCase(name="batch4", B=4, S=3, seed=789, description="Multi batch (B=4, S=3)"),
    "batch8": TestCase(name="batch8", B=8, S=16, seed=444, description="Large input (B=8, S=16)"),
    "edge1": TestCase(name="edge1", B=1, S=1, seed=333, description="Minimal (B=1, S=1)"),
    "long": TestCase(name="long", B=1, S=32, seed=111, description="Long sequence (B=1, S=32)"),
}


def read_binary_f32(path: str) -> np.ndarray:
    """Read raw float32 binary file"""
    with open(path, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    values = struct.unpack(f'{count}f', data)
    return np.array(values, dtype=np.float32)


def run_command(cmd: str, cwd: str = None, timeout: int = 60) -> Tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def build_cpp(project_root: str) -> bool:
    """Build C++ inference engine"""
    cpp_dir = os.path.join(project_root, 'cpp')
    build_dir = os.path.join(cpp_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)
    
    ret, out, err = run_command('cmake ..', cwd=build_dir)
    if ret != 0:
        print(f'CMake failed: {err}')
        return False
        
    ret, out, err = run_command('make -j4', cwd=build_dir)
    if ret != 0:
        print(f'Make failed: {err}')
        return False
    return True


def run_pytorch_inference(test_case: TestCase, project_root: str, dump_dir: str) -> Tuple[float, str]:
    """Run PyTorch inference and return (elapsed_ms, logits_path)"""
    import importlib
    
    # Set environment
    os.environ['DUMP_DIR'] = dump_dir
    os.environ['JSON_PATH'] = os.path.join(dump_dir, 'minimind.json')
    os.environ['B'] = str(test_case.B)
    os.environ['S'] = str(test_case.S)
    os.environ['SEED'] = str(test_case.seed)
    
    # Generate weights
    from python.export import minimind_dumper
    importlib.reload(minimind_dumper)
    minimind_dumper.main()
    
    # Run inference
    from python.inference import minimind_forward
    importlib.reload(minimind_forward)
    minimind_forward.main()
    
    # Read timing
    timing_path = os.path.join(dump_dir, 'timing_torch.json')
    with open(timing_path) as f:
        timing = json.load(f)
    
    return timing.get('elapsed_ms', 0.0), os.path.join(dump_dir, 'logits_torch.npy')


def run_unified_backend(
    backend: str,
    json_path: str, 
    dump_dir: str,
    project_root: str
) -> BackendResult:
    """Run unified binary with specific backend"""
    unified_bin = os.path.join(project_root, 'cpp', 'build', 'minimind_unified')
    
    if not os.path.exists(unified_bin):
        return BackendResult(
            backend=backend,
            elapsed_ms=0,
            max_logit=0, min_logit=0, mean_logit=0,
            success=False,
            error="minimind_unified not found"
        )
    
    cmd = f'{unified_bin} {json_path} {dump_dir} "{backend}"'
    ret, out, err = run_command(cmd, cwd=project_root)
    
    if ret != 0:
        return BackendResult(
            backend=backend,
            elapsed_ms=0,
            max_logit=0, min_logit=0, mean_logit=0,
            success=False,
            error=err or "Unknown error"
        )
    
    # Parse output
    elapsed_ms = 0.0
    match = re.search(r'\[Timing\]\s*([\d.]+)\s*ms', out)
    if match:
        elapsed_ms = float(match.group(1))
    
    max_logit = min_logit = mean_logit = 0.0
    match = re.search(r'\[Logits\]\s*Min:\s*([-\d.]+),\s*Max:\s*([-\d.]+),\s*Mean:\s*([-\d.]+)', out)
    if match:
        min_logit = float(match.group(1))
        max_logit = float(match.group(2))
        mean_logit = float(match.group(3))
    
    return BackendResult(
        backend=backend,
        elapsed_ms=elapsed_ms,
        max_logit=max_logit,
        min_logit=min_logit,
        mean_logit=mean_logit,
        success=True
    )


def compare_logits(
    pytorch_path: str,
    cpp_path: str,
    shape: Tuple[int, ...],
    atol: float = 1e-3,
    rtol: float = 1e-3
) -> Tuple[bool, float, float]:
    """Compare logits, return (is_close, max_diff, mean_diff)"""
    try:
        torch_logits = np.load(pytorch_path, allow_pickle=True).astype(np.float32)
        cpp_logits = read_binary_f32(cpp_path).reshape(shape)
        
        diff = np.abs(torch_logits - cpp_logits)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        is_close = np.allclose(torch_logits, cpp_logits, rtol=rtol, atol=atol)
        
        return is_close, max_diff, mean_diff
    except Exception as e:
        return False, float('inf'), float('inf')


def run_benchmark(
    test_case: TestCase,
    backends: List[str],
    project_root: str,
    dump_dir: str,
    verbose: bool = False
) -> BenchmarkResult:
    """Run complete benchmark for a test case"""
    # Run PyTorch
    print(f"  Running PyTorch inference...", end=" ", flush=True)
    pytorch_ms, pytorch_logits = run_pytorch_inference(test_case, project_root, dump_dir)
    print(f"{pytorch_ms:.2f}ms")
    
    result = BenchmarkResult(
        test_case=test_case,
        pytorch_ms=pytorch_ms,
        pytorch_logits_path=pytorch_logits
    )
    
    json_path = os.path.join(dump_dir, 'minimind.json')
    
    # Run each backend
    for backend in backends:
        print(f"  Running {backend}...", end=" ", flush=True)
        br = run_unified_backend(backend, json_path, dump_dir, project_root)
        result.backend_results.append(br)
        
        if br.success:
            print(f"{br.elapsed_ms:.2f}ms")
        else:
            print(f"FAILED: {br.error}")
    
    return result


def print_results(results: List[BenchmarkResult], backends: List[str]):
    """Print formatted benchmark results"""
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    
    # Header
    header = f"{'Test Case':<12} {'B*S':<6} {'PyTorch':>10}"
    for b in backends:
        header += f" {b:>14}"
    print(header)
    print("-" * 120)
    
    # Results
    total_torch = 0.0
    total_backends = {b: 0.0 for b in backends}
    
    for r in results:
        tc = r.test_case
        row = f"{tc.name:<12} {tc.B * tc.S:<6} {r.pytorch_ms:>10.2f}ms"
        total_torch += r.pytorch_ms
        
        for backend in backends:
            br = next((x for x in r.backend_results if x.backend == backend), None)
            if br and br.success:
                row += f" {br.elapsed_ms:>10.2f}ms"
                total_backends[backend] += br.elapsed_ms
            else:
                row += f" {'FAIL':>10}"
        
        print(row)
    
    # Totals
    print("-" * 120)
    total_row = f"{'TOTAL':<12} {'':<6} {total_torch:>10.2f}ms"
    for backend in backends:
        if total_backends[backend] > 0:
            total_row += f" {total_backends[backend]:>10.2f}ms"
        else:
            total_row += f" {'N/A':>10}"
    print(total_row)
    
    # Speedups
    print("\n" + "=" * 120)
    print("SPEEDUP vs PyTorch")
    print("=" * 120)
    
    header = f"{'Test Case':<12} {'B*S':<6}"
    for b in backends:
        header += f" {b:>14}"
    print(header)
    print("-" * 120)
    
    for r in results:
        tc = r.test_case
        row = f"{tc.name:<12} {tc.B * tc.S:<6}"
        speedups = r.get_speedups()
        
        for backend in backends:
            if backend in speedups:
                s = speedups[backend]
                indicator = "🚀" if s > 2.0 else ("⬆️" if s > 1.0 else "⬇️")
                row += f" {s:>10.2f}x {indicator}"
            else:
                row += f" {'N/A':>12}"
        
        print(row)
    
    # vs Baseline speedups
    if "baseline" in backends:
        print("\n" + "=" * 120)
        print("SPEEDUP vs Baseline")
        print("=" * 120)
        
        header = f"{'Test Case':<12} {'B*S':<6}"
        for b in backends:
            if b != "baseline":
                header += f" {b:>14}"
        print(header)
        print("-" * 120)
        
        for r in results:
            tc = r.test_case
            row = f"{tc.name:<12} {tc.B * tc.S:<6}"
            vs_baseline = r.get_vs_baseline()
            
            for backend in backends:
                if backend == "baseline":
                    continue
                if backend in vs_baseline:
                    s = vs_baseline[backend]
                    indicator = "🚀" if s > 1.5 else ("⬆️" if s > 1.0 else "⬇️")
                    row += f" {s:>10.2f}x {indicator}"
                else:
                    row += f" {'N/A':>12}"
            
            print(row)
    
    print("\n" + "=" * 120)
    print("Legend: 🚀 = >2x faster  ⬆️ = faster  ⬇️ = slower")
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description='Multi-Backend Benchmark Tool')
    parser.add_argument('--backends', '-b', default='baseline,simd',
                        help='Backends to test (comma-separated or "all")')
    parser.add_argument('--test-case', '-t', default='basic',
                        choices=list(TEST_CASES.keys()) + ['all'],
                        help='Test case to run')
    parser.add_argument('--dump-dir', default='dump_minimind', help='Dump directory')
    parser.add_argument('--skip-build', action='store_true', help='Skip C++ build')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dump_dir = os.path.join(project_root, args.dump_dir)
    os.makedirs(dump_dir, exist_ok=True)
    
    # Parse backends
    if args.backends.lower() == 'all':
        backends = ALL_BACKENDS
    else:
        backends = [b.strip() for b in args.backends.split(',')]
    
    # Parse test cases
    if args.test_case == 'all':
        test_cases = list(TEST_CASES.values())
    else:
        test_cases = [TEST_CASES[args.test_case]]
    
    print("=" * 120)
    print(f"CelerInfer Multi-Backend Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)
    print(f"Backends: {', '.join(backends)}")
    print(f"Test cases: {', '.join(tc.name for tc in test_cases)}")
    print()
    
    # Build
    if not args.skip_build:
        print("[Build] Compiling C++ inference engine...")
        if not build_cpp(project_root):
            print("Build failed!")
            return 1
        print("[Build] Success!\n")
    
    # Run benchmarks
    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {tc.name}: {tc.description}")
        result = run_benchmark(tc, backends, project_root, dump_dir, args.verbose)
        results.append(result)
        print()
    
    # Print results
    print_results(results, backends)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
DiT Multi-Backend Benchmark Tool

用于对 DiT (Diffusion Transformer) 模型进行多平台/多规模的性能基准测试。

Usage:
    python -m python.tools.dit_benchmark [options]

Examples:
    # 运行默认配置
    python -m python.tools.dit_benchmark
    
    # 指定后端
    python -m python.tools.dit_benchmark --backends baseline simd extreme
    
    # 自定义模型规模
    python -m python.tools.dit_benchmark --hidden-dim 256 --encoder-depth 4 --decoder-depth 8
    
    # 保存结果
    python -m python.tools.dit_benchmark --output results.json
"""

import argparse
import json
import os
import sys
import time
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from python.core.dit import ModelConfig, Diffusion_Planner


# ============================================================================
# Predefined Model Scales
# ============================================================================
PREDEFINED_SCALES = {
    "tiny": {
        "hidden_dim": 96,
        "encoder_depth": 2,
        "decoder_depth": 2,
        "num_heads": 4,
        "description": "Tiny model for quick tests (~1M params)"
    },
    "small": {
        "hidden_dim": 128,
        "encoder_depth": 3,
        "decoder_depth": 4,
        "num_heads": 4,
        "description": "Small model (~3M params)"
    },
    "base": {
        "hidden_dim": 192,
        "encoder_depth": 4,
        "decoder_depth": 6,
        "num_heads": 6,
        "description": "Base model (~8M params)"
    },
    "medium": {
        "hidden_dim": 256,
        "encoder_depth": 6,
        "decoder_depth": 8,
        "num_heads": 8,
        "description": "Medium model (~20M params)"
    },
    "large": {
        "hidden_dim": 384,
        "encoder_depth": 8,
        "decoder_depth": 10,
        "num_heads": 8,
        "description": "Large model (~50M params)"
    },
    "xlarge": {
        "hidden_dim": 512,
        "encoder_depth": 10,
        "decoder_depth": 12,
        "num_heads": 8,
        "description": "XLarge model (~100M params)"
    }
}


@dataclass
class DiTBenchmarkConfig:
    """DiT benchmark configuration"""
    # Model architecture
    hidden_dim: int = 192
    encoder_depth: int = 4
    decoder_depth: int = 6
    num_heads: int = 6
    
    # Input dimensions
    future_len: int = 80
    time_len: int = 21
    agent_num: int = 32
    lane_num: int = 50
    static_objects_num: int = 5
    predicted_neighbor_num: int = 5
    
    # Benchmark settings
    batch_size: int = 1
    warmup_runs: int = 3
    benchmark_runs: int = 10
    device: str = "cpu"


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    backend: str
    scale: str
    batch_size: int
    hidden_dim: int
    encoder_depth: int
    decoder_depth: int
    param_count: int
    mean_latency_ms: float
    std_latency_ms: float
    throughput: float  # samples/sec
    success: bool
    error_msg: str = ""


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_config(cfg: DiTBenchmarkConfig) -> ModelConfig:
    """Create ModelConfig from benchmark config"""
    return ModelConfig(
        future_len=cfg.future_len,
        time_len=cfg.time_len,
        agent_state_dim=11,  # 默认值，与 dit.py 一致
        agent_num=cfg.agent_num,
        static_objects_state_dim=10,  # 默认值，与 dit.py 一致
        static_objects_num=cfg.static_objects_num,
        lane_len=20,
        lane_state_dim=12,  # 默认值，与 dit.py 一致
        lane_num=cfg.lane_num,
        route_len=20,  # 默认值
        route_state_dim=12,  # 默认值，与 dit.py 一致
        route_num=25,  # 默认值
        encoder_depth=cfg.encoder_depth,
        decoder_depth=cfg.decoder_depth,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        predicted_neighbor_num=cfg.predicted_neighbor_num,
        diffusion_model_type='x_start',
        device='cpu',
    )


def create_dummy_inputs(cfg: DiTBenchmarkConfig, device: str) -> Dict[str, torch.Tensor]:
    """Create dummy inputs for benchmarking (matching dit.py format)"""
    B = cfg.batch_size
    P = 1 + cfg.predicted_neighbor_num
    
    # 使用与 dit.py 中 create_dummy_inputs 一致的格式
    return {
        "ego_current_state": torch.randn(B, 10, device=device),  # 固定维度 10
        "neighbor_agents_past": torch.randn(B, cfg.agent_num, cfg.time_len, 11, device=device),  # agent_state_dim=11
        "static_objects": torch.randn(B, cfg.static_objects_num, 10, device=device),  # static_objects_state_dim=10
        "lanes": torch.randn(B, cfg.lane_num, 20, 12, device=device),  # lane_len=20, lane_state_dim=12
        "lanes_speed_limit": torch.randn(B, cfg.lane_num, 1, device=device),  # 注意这里是 [B, lane_num, 1]
        "lanes_has_speed_limit": torch.randint(0, 2, (B, cfg.lane_num, 1), dtype=torch.bool, device=device),
        "route_lanes": torch.randn(B, 25, 20, 12, device=device),  # route_num=25, route_len=20, route_state_dim=12
        "sampled_trajectories": torch.randn(B, P, cfg.future_len + 1, 4, device=device),  # 注意4D！
        "diffusion_time": torch.rand(B, device=device),
    }


def benchmark_pytorch(cfg: DiTBenchmarkConfig) -> Tuple[float, float]:
    """Benchmark PyTorch implementation"""
    from python.core.dit import create_dummy_inputs as dit_create_inputs
    
    model_cfg = create_model_config(cfg)
    model = Diffusion_Planner(model_cfg)
    model.eval()
    
    if cfg.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"
    
    # 使用官方的输入创建函数
    inputs = dit_create_inputs(model_cfg, batch_size=cfg.batch_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(cfg.warmup_runs):
            # 使用模型的 forward 方法
            _ = model(inputs)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(cfg.benchmark_runs):
            start = time.perf_counter()
            # 使用模型的 forward 方法
            _ = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    return np.mean(latencies), np.std(latencies)


def export_model_for_cpp(cfg: DiTBenchmarkConfig, output_path: str) -> int:
    """Export model weights to JSON for C++ inference"""
    model_cfg = create_model_config(cfg)
    model = Diffusion_Planner(model_cfg)
    model.eval()
    
    param_count = count_parameters(model)
    
    # Import dumper and its input creator
    from python.export.dit_dumper import DiTDumper
    from python.core.dit import create_dummy_inputs as dit_create_inputs
    
    # Create inputs using the official function from dit.py
    inputs = dit_create_inputs(model_cfg, batch_size=cfg.batch_size, device='cpu')
    
    # Get the output directory and filename
    output_dir = os.path.dirname(output_path) or "."
    
    dumper = DiTDumper()
    dumper.dump(model, model_cfg, inputs, output_dir)
    
    # Load the JSON and make sure inputs are in correct format for C++
    json_path = os.path.join(output_dir, "dit.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    import base64
    
    def tensor_to_json(t: torch.Tensor) -> dict:
        arr = t.detach().cpu().numpy().astype(np.float32)
        return {
            "shape": list(arr.shape),
            "dtype": "float32",
            "data": base64.b64encode(arr.tobytes()).decode('ascii')
        }
    
    # Ensure inputs are in expected format for C++ (flatten sampled_trajectories)
    if "inputs" not in data:
        data["inputs"] = {}
    
    # For C++, we flatten the sampled_trajectories to [B, P, (future_len+1)*4]
    sampled = inputs["sampled_trajectories"]
    B, P, T, D = sampled.shape
    sampled_flat = sampled.view(B, P, T * D)
    data["inputs"]["sampled_trajectories"] = tensor_to_json(sampled_flat)
    data["inputs"]["diffusion_time"] = tensor_to_json(inputs["diffusion_time"])
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    return param_count


def benchmark_cpp_backend(backend: str, json_path: str, dump_dir: str, 
                          build_dir: str) -> Tuple[Optional[float], str]:
    """Benchmark C++ backend"""
    # Map backend name to executable name
    backend_exe_map = {
        "baseline": "dit",
        "simd": "dit_simd",
        "extreme": "dit_extreme",
        "cuda": "dit_cuda",
        "cuda_fp16": "dit_cuda_fp16",
    }
    exe_name = backend_exe_map.get(backend, f"dit_{backend}")
    exe_path = os.path.join(build_dir, exe_name)
    
    if not os.path.exists(exe_path):
        return None, f"Executable not found: {exe_path}"
    
    try:
        result = subprocess.run(
            [exe_path, json_path, dump_dir],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return None, f"Exit code {result.returncode}: {result.stderr}"
        
        # Parse timing from output
        for line in result.stdout.split('\n'):
            if '[Timing]' in line and 'Forward pass' in line:
                # Format: "[Timing] Forward pass: 12.34ms"
                parts = line.split(':')
                if len(parts) >= 2:
                    time_str = parts[-1].strip().replace('ms', '')
                    return float(time_str), ""
        
        return None, "Could not parse timing from output"
        
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def run_benchmark(
    scales: List[str],
    backends: List[str],
    cfg: DiTBenchmarkConfig,
    cpp_build_dir: str
) -> List[BenchmarkResult]:
    """Run full benchmark suite"""
    results = []
    
    for scale in scales:
        print(f"\n{'='*60}")
        print(f"Scale: {scale}")
        print(f"{'='*60}")
        
        # Update config for this scale
        if scale in PREDEFINED_SCALES:
            scale_cfg = PREDEFINED_SCALES[scale]
            cfg.hidden_dim = scale_cfg["hidden_dim"]
            cfg.encoder_depth = scale_cfg["encoder_depth"]
            cfg.decoder_depth = scale_cfg["decoder_depth"]
            cfg.num_heads = scale_cfg["num_heads"]
            print(f"  {scale_cfg['description']}")
        
        print(f"  hidden_dim={cfg.hidden_dim}, encoder_depth={cfg.encoder_depth}, "
              f"decoder_depth={cfg.decoder_depth}, num_heads={cfg.num_heads}")
        
        # Create temporary directory for this scale
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "dit.json")
            
            # Export model
            print(f"  Exporting model...")
            try:
                param_count = export_model_for_cpp(cfg, json_path)
                print(f"  Parameters: {param_count:,}")
            except Exception as e:
                print(f"  Export failed: {e}")
                continue
            
            for backend in backends:
                print(f"\n  Backend: {backend}")
                
                if backend == "pytorch":
                    try:
                        mean_lat, std_lat = benchmark_pytorch(cfg)
                        throughput = 1000.0 / mean_lat * cfg.batch_size
                        results.append(BenchmarkResult(
                            backend=backend,
                            scale=scale,
                            batch_size=cfg.batch_size,
                            hidden_dim=cfg.hidden_dim,
                            encoder_depth=cfg.encoder_depth,
                            decoder_depth=cfg.decoder_depth,
                            param_count=param_count,
                            mean_latency_ms=mean_lat,
                            std_latency_ms=std_lat,
                            throughput=throughput,
                            success=True
                        ))
                        print(f"    Latency: {mean_lat:.2f} ± {std_lat:.2f} ms")
                        print(f"    Throughput: {throughput:.2f} samples/sec")
                    except Exception as e:
                        results.append(BenchmarkResult(
                            backend=backend,
                            scale=scale,
                            batch_size=cfg.batch_size,
                            hidden_dim=cfg.hidden_dim,
                            encoder_depth=cfg.encoder_depth,
                            decoder_depth=cfg.decoder_depth,
                            param_count=param_count,
                            mean_latency_ms=0,
                            std_latency_ms=0,
                            throughput=0,
                            success=False,
                            error_msg=str(e)
                        ))
                        print(f"    FAILED: {e}")
                else:
                    latency, error = benchmark_cpp_backend(
                        backend, json_path, tmpdir, cpp_build_dir
                    )
                    
                    if latency is not None:
                        throughput = 1000.0 / latency * cfg.batch_size
                        results.append(BenchmarkResult(
                            backend=backend,
                            scale=scale,
                            batch_size=cfg.batch_size,
                            hidden_dim=cfg.hidden_dim,
                            encoder_depth=cfg.encoder_depth,
                            decoder_depth=cfg.decoder_depth,
                            param_count=param_count,
                            mean_latency_ms=latency,
                            std_latency_ms=0,
                            throughput=throughput,
                            success=True
                        ))
                        print(f"    Latency: {latency:.2f} ms")
                        print(f"    Throughput: {throughput:.2f} samples/sec")
                    else:
                        results.append(BenchmarkResult(
                            backend=backend,
                            scale=scale,
                            batch_size=cfg.batch_size,
                            hidden_dim=cfg.hidden_dim,
                            encoder_depth=cfg.encoder_depth,
                            decoder_depth=cfg.decoder_depth,
                            param_count=param_count,
                            mean_latency_ms=0,
                            std_latency_ms=0,
                            throughput=0,
                            success=False,
                            error_msg=error
                        ))
                        print(f"    FAILED: {error}")
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary table"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Group by scale
    scales = sorted(set(r.scale for r in results), 
                   key=lambda s: list(PREDEFINED_SCALES.keys()).index(s) if s in PREDEFINED_SCALES else 999)
    backends = sorted(set(r.backend for r in results))
    
    # Header
    header = f"{'Scale':<10} {'Params':>12}"
    for backend in backends:
        header += f" {backend:>12}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for scale in scales:
        scale_results = [r for r in results if r.scale == scale]
        if not scale_results:
            continue
        
        param_count = scale_results[0].param_count
        row = f"{scale:<10} {param_count:>10,}"
        
        for backend in backends:
            br = next((r for r in scale_results if r.backend == backend), None)
            if br and br.success:
                row += f" {br.mean_latency_ms:>10.2f}ms"
            else:
                row += f" {'N/A':>12}"
        
        print(row)
    
    print("-" * len(header))
    
    # Speedup vs PyTorch
    print("\nSpeedup vs PyTorch:")
    print("-" * 50)
    for scale in scales:
        pytorch_result = next((r for r in results if r.scale == scale and r.backend == "pytorch" and r.success), None)
        if not pytorch_result:
            continue
        
        row = f"{scale:<10}"
        for backend in backends:
            if backend == "pytorch":
                continue
            br = next((r for r in results if r.scale == scale and r.backend == backend and r.success), None)
            if br:
                speedup = pytorch_result.mean_latency_ms / br.mean_latency_ms
                row += f" {backend}: {speedup:.2f}x"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="DiT Multi-Backend Benchmark")
    
    # Scale selection
    parser.add_argument("--scales", nargs="+", 
                       default=["tiny", "small", "base"],
                       choices=list(PREDEFINED_SCALES.keys()) + ["custom"],
                       help="Model scales to benchmark")
    
    # Backend selection
    parser.add_argument("--backends", nargs="+",
                       default=["pytorch", "baseline", "simd", "extreme", "cuda", "cuda_fp16"],
                       help="Backends to benchmark")
    
    # Custom model config
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    
    # Benchmark settings
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    
    # Build directory
    parser.add_argument("--build-dir", type=str, 
                       default=str(project_root / "cpp" / "build"),
                       help="C++ build directory")
    
    # Output
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Create config
    cfg = DiTBenchmarkConfig(
        hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        device=args.device,
    )
    
    print("DiT Multi-Backend Benchmark")
    print("="*60)
    print(f"Scales: {args.scales}")
    print(f"Backends: {args.backends}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"C++ build dir: {args.build_dir}")
    
    # Run benchmarks
    results = run_benchmark(
        scales=args.scales,
        backends=args.backends,
        cfg=cfg,
        cpp_build_dir=args.build_dir
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

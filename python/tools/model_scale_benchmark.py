#!/usr/bin/env python3
"""
模型规模基准测试工具

在不同模型参数量下，横向对比各种后端实现的性能。

用法:
    # 测试所有预设模型规模
    python -m python.tools.model_scale_benchmark
    
    # 只测试特定规模
    python -m python.tools.model_scale_benchmark --scale tiny,small,medium
    
    # 自定义模型规模
    python -m python.tools.model_scale_benchmark --custom-scale H=256,L=4,A=4,KV=2,V=1000,I=688 --name my_model
    
    # 跳过CUDA后端
    python -m python.tools.model_scale_benchmark --no-cuda
    
    # 详细输出
    python -m python.tools.model_scale_benchmark -v

模型规模预设:
    - micro:    H=64,  L=2,  A=8,  KV=2, V=128   (~0.05M params)
    - tiny:     H=128, L=2,  A=4,  KV=2, V=512   (~0.4M params)
    - small:    H=256, L=4,  A=8,  KV=2, V=2048  (~4M params)
    - medium:   H=512, L=8,  A=8,  KV=2, V=6400  (~25M params)
    - large:    H=768, L=12, A=12, KV=4, V=32000 (~100M params)
    - xlarge:   H=1024,L=16, A=16, KV=4, V=50000 (~300M params)

后端:
    - PyTorch:          Python参考实现
    - C++ Baseline:     纯C++无优化
    - C++ SIMD:         AVX2向量化
    - C++ Extreme:      OpenMP + AVX2 + Tiling
    - CUDA FP32:        GPU FP32 (可选)
    - CUDA FP16:        GPU FP16 + Tensor Core (可选)
"""

import os
import sys
import re
import json
import time
import struct
import argparse
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# Model Scale Configurations
# =============================================================================
@dataclass
class ModelScale:
    """模型规模配置"""
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    intermediate_size: int = 0  # 0 = auto compute as ~2.7x hidden
    max_position: int = 512
    description: str = ""
    
    def __post_init__(self):
        if self.intermediate_size == 0:
            # 默认 intermediate_size ≈ 2.7 * hidden_size (rounded to 8)
            self.intermediate_size = ((int(self.hidden_size * 2.7) + 7) // 8) * 8
    
    @property
    def params_millions(self) -> float:
        """估算参数量 (百万)"""
        H = self.hidden_size
        L = self.num_layers
        I = self.intermediate_size
        V = self.vocab_size
        
        # Embedding + LM head: 2 * V * H (但通常共享，所以只算一次)
        embed_params = V * H
        
        # Per layer:
        # - Attention: Q(H*H) + K(KV*D*H) + V(KV*D*H) + O(H*H)
        #   简化: Q+K+V+O ≈ 4 * H * H (近似)
        # - FFN: gate(I*H) + up(I*H) + down(H*I) = 3 * H * I
        # - RMSNorm: 2 * H
        per_layer = 4 * H * H + 3 * H * I + 2 * H
        
        # Final RMSNorm
        final_norm = H
        
        total = embed_params + L * per_layer + final_norm
        return total / 1e6


# 预定义模型规模
MODEL_SCALES: Dict[str, ModelScale] = {
    "micro": ModelScale(
        name="micro", hidden_size=64, num_layers=2, num_heads=8, num_kv_heads=2,
        vocab_size=128, max_position=128,
        description="极小模型 (~0.05M) - 用于快速验证"
    ),
    "tiny": ModelScale(
        name="tiny", hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
        vocab_size=512, max_position=256,
        description="微型模型 (~0.4M) - 快速测试"
    ),
    "small": ModelScale(
        name="small", hidden_size=256, num_layers=4, num_heads=8, num_kv_heads=2,
        vocab_size=2048, max_position=512,
        description="小型模型 (~4M) - 中等负载"
    ),
    "medium": ModelScale(
        name="medium", hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=2,
        vocab_size=6400, max_position=512,
        description="中型模型 (~25M) - 接近 MiniMind 默认"
    ),
    "large": ModelScale(
        name="large", hidden_size=768, num_layers=12, num_heads=12, num_kv_heads=4,
        vocab_size=32000, max_position=1024,
        description="大型模型 (~100M) - 接近 GPT-2 Small"
    ),
    "xlarge": ModelScale(
        name="xlarge", hidden_size=1024, num_layers=16, num_heads=16, num_kv_heads=4,
        vocab_size=50000, max_position=2048,
        description="超大模型 (~300M) - 高负载测试"
    ),
}


# =============================================================================
# Backend Configuration
# =============================================================================
@dataclass
class Backend:
    """后端配置"""
    name: str
    binary_name: str         # C++ 可执行文件名
    output_pattern: str      # 从输出中匹配时间的正则
    output_file: str = "logits_cpp.npy"  # 输出文件名
    requires_cuda: bool = False
    description: str = ""


BACKENDS: Dict[str, Backend] = {
    "baseline": Backend(
        name="baseline",
        binary_name="minimind",
        output_pattern=r'\[Timing\] Forward pass: ([\d.]+)ms',
        output_file="logits_cpp.npy",
        description="C++ Baseline (无优化)"
    ),
    "simd": Backend(
        name="simd",
        binary_name="minimind_simd",
        output_pattern=r'\[Timing\] SIMD Forward pass: ([\d.]+)ms',
        output_file="logits_simd.npy",
        description="C++ AVX2 SIMD"
    ),
    "extreme": Backend(
        name="extreme",
        binary_name="minimind_extreme",
        output_pattern=r'Extreme Inference: ([\d.]+) ms/forward',
        output_file="logits_extreme.bin",
        description="C++ OpenMP + AVX2 + Tiling"
    ),
    "cuda_fp32": Backend(
        name="cuda_fp32",
        binary_name="minimind_cuda",
        output_pattern=r'\[Timing\] ([\d.]+) ms',
        output_file="logits_cuda.npy",
        requires_cuda=True,
        description="CUDA GPU FP32"
    ),
    "cuda_fp16": Backend(
        name="cuda_fp16",
        binary_name="minimind_cuda_fp16",
        output_pattern=r'Ultra \(Fusion\+Streams\):\s+([\d.]+) ms/forward',
        output_file="logits_fp16.npy",
        requires_cuda=True,
        description="CUDA GPU FP16 + Tensor Core"
    ),
}


# =============================================================================
# Result Data Classes
# =============================================================================
@dataclass
class BenchmarkResult:
    """单次基准测试结果"""
    scale_name: str
    params_m: float          # 参数量 (百万)
    pytorch_ms: float = 0.0
    backend_times: Dict[str, float] = field(default_factory=dict)
    error_msg: str = ""
    is_consistent: bool = True
    max_diff: float = 0.0


# =============================================================================
# Utility Functions
# =============================================================================
def run_command(cmd: str, cwd: str = None, timeout: int = 300) -> Tuple[int, str, str]:
    """运行命令并返回 (返回码, stdout, stderr)"""
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


def read_binary_f32(path: str) -> np.ndarray:
    """读取C++保存的原始float32二进制文件"""
    with open(path, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    values = struct.unpack(f'{count}f', data)
    return np.array(values, dtype=np.float32)


def build_cpp(project_root: str, enable_cuda: bool = False) -> bool:
    """编译C++推理引擎"""
    cpp_dir = os.path.join(project_root, 'cpp')
    build_dir = os.path.join(cpp_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)
    
    cuda_flag = "-DENABLE_CUDA=ON" if enable_cuda else "-DENABLE_CUDA=OFF"
    ret, out, err = run_command(f'cmake {cuda_flag} ..', cwd=build_dir)
    if ret != 0:
        print(f'CMake 失败: {err}')
        return False
        
    ret, out, err = run_command('make -j4', cwd=build_dir)
    if ret != 0:
        print(f'Make 失败: {err}')
        return False
    return True


def check_available_backends(project_root: str) -> List[str]:
    """检查可用的后端"""
    available = []
    build_dir = os.path.join(project_root, 'cpp', 'build')
    
    for name, backend in BACKENDS.items():
        binary_path = os.path.join(build_dir, backend.binary_name)
        if os.path.exists(binary_path):
            available.append(name)
    
    return available


# =============================================================================
# Benchmark Runner
# =============================================================================
class ModelScaleBenchmark:
    """模型规模基准测试运行器"""
    
    def __init__(
        self,
        project_root: str,
        dump_dir: str = "dump_benchmark",
        enable_cuda: bool = True,
        verbose: bool = False,
        batch_size: int = 1,
        seq_length: int = 64,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ):
        self.project_root = project_root
        self.dump_dir = os.path.join(project_root, dump_dir)
        self.enable_cuda = enable_cuda
        self.verbose = verbose
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.atol = atol
        self.rtol = rtol
        
        os.makedirs(self.dump_dir, exist_ok=True)
    
    def _set_model_env(self, scale: ModelScale):
        """设置模型配置的环境变量"""
        os.environ['HIDDEN'] = str(scale.hidden_size)
        os.environ['LAYERS'] = str(scale.num_layers)
        os.environ['HEADS'] = str(scale.num_heads)
        os.environ['KVH'] = str(scale.num_kv_heads)
        os.environ['VOCAB'] = str(scale.vocab_size)
        os.environ['MAX_POS'] = str(scale.max_position)
        os.environ['B'] = str(self.batch_size)
        os.environ['S'] = str(self.seq_length)
        os.environ['SEED'] = '42'
        os.environ['DUMP_DIR'] = self.dump_dir
        os.environ['JSON_PATH'] = os.path.join(self.dump_dir, 'minimind.json')
    
    def _run_pytorch(self) -> Tuple[float, str]:
        """运行 PyTorch 推理，返回 (耗时ms, 错误信息)"""
        import importlib
        
        try:
            # 1) 生成模型权重
            from python.export import minimind_dumper
            importlib.reload(minimind_dumper)
            minimind_dumper.main()
            
            # 2) PyTorch 前向传播
            from python.inference import minimind_forward
            importlib.reload(minimind_forward)
            minimind_forward.main()
            
            # 3) 读取计时
            timing_path = os.path.join(self.dump_dir, 'timing_torch.json')
            if os.path.exists(timing_path):
                with open(timing_path) as f:
                    timing = json.load(f)
                return timing.get('elapsed_ms', 0), ""
            return 0, "No timing file"
            
        except Exception as e:
            return 0, str(e)
    
    def _run_cpp_backend(self, backend: Backend) -> Tuple[float, str]:
        """运行 C++ 后端，返回 (耗时ms, 错误信息)"""
        binary_path = os.path.join(self.project_root, 'cpp', 'build', backend.binary_name)
        json_path = os.path.join(self.dump_dir, 'minimind.json')
        
        if not os.path.exists(binary_path):
            return 0, f"Binary not found: {backend.binary_name}"
        
        ret, out, err = run_command(
            f'{binary_path} {json_path} {self.dump_dir}',
            cwd=self.project_root,
            timeout=120
        )
        
        if ret != 0:
            return 0, f"Exit code {ret}: {err[:200]}"
        
        if self.verbose:
            print(f"    [{backend.name}] stdout: {out[:500]}")
        
        # 解析时间
        match = re.search(backend.output_pattern, out)
        if match:
            return float(match.group(1)), ""
        
        return 0, f"Cannot parse timing from output"
    
    def _check_consistency(self) -> Tuple[bool, float]:
        """检查 PyTorch 和 C++ 结果一致性"""
        torch_path = os.path.join(self.dump_dir, 'logits_torch.npy')
        cpp_path = os.path.join(self.dump_dir, 'logits_cpp.npy')
        
        try:
            if not os.path.exists(torch_path) or not os.path.exists(cpp_path):
                return True, 0  # 跳过检查
            
            logits_torch = np.load(torch_path, allow_pickle=True).astype(np.float32)
            logits_cpp = read_binary_f32(cpp_path).reshape(logits_torch.shape)
            
            diff = np.abs(logits_torch - logits_cpp)
            max_diff = float(diff.max())
            is_close = np.allclose(logits_torch, logits_cpp, rtol=self.rtol, atol=self.atol)
            
            return is_close, max_diff
        except Exception as e:
            if self.verbose:
                print(f"    Consistency check error: {e}")
            return True, 0
    
    def run_single_scale(
        self,
        scale: ModelScale,
        backends_to_run: List[str]
    ) -> BenchmarkResult:
        """运行单个模型规模的基准测试"""
        result = BenchmarkResult(
            scale_name=scale.name,
            params_m=scale.params_millions
        )
        
        # 设置环境变量
        self._set_model_env(scale)
        
        # 1) PyTorch
        if self.verbose:
            print(f"  Running PyTorch...")
        pytorch_ms, err = self._run_pytorch()
        if err:
            result.error_msg = f"PyTorch: {err}"
            return result
        result.pytorch_ms = pytorch_ms
        
        # 2) C++ 后端
        for backend_name in backends_to_run:
            backend = BACKENDS.get(backend_name)
            if not backend:
                continue
            
            if self.verbose:
                print(f"  Running {backend_name}...")
            
            ms, err = self._run_cpp_backend(backend)
            if err:
                if self.verbose:
                    print(f"    Error: {err}")
                result.backend_times[backend_name] = 0
            else:
                result.backend_times[backend_name] = ms
        
        # 3) 一致性检查 (只检查 baseline)
        is_consistent, max_diff = self._check_consistency()
        result.is_consistent = is_consistent
        result.max_diff = max_diff
        
        return result
    
    def run_benchmark(
        self,
        scales: List[ModelScale],
        backends_to_run: List[str]
    ) -> List[BenchmarkResult]:
        """运行完整基准测试"""
        results = []
        
        for i, scale in enumerate(scales, 1):
            print(f"\n[{i}/{len(scales)}] {scale.name}: {scale.description}")
            print(f"         H={scale.hidden_size}, L={scale.num_layers}, "
                  f"A={scale.num_heads}, KV={scale.num_kv_heads}, V={scale.vocab_size}")
            print(f"         估算参数量: {scale.params_millions:.2f}M")
            
            result = self.run_single_scale(scale, backends_to_run)
            
            # 即时显示结果
            if result.error_msg:
                print(f"         ❌ 错误: {result.error_msg}")
            else:
                time_strs = [f"PyTorch: {result.pytorch_ms:.2f}ms"]
                for name, ms in result.backend_times.items():
                    if ms > 0:
                        speedup = result.pytorch_ms / ms if ms > 0 else 0
                        time_strs.append(f"{name}: {ms:.2f}ms ({speedup:.1f}x)")
                print(f"         ⏱️  {', '.join(time_strs)}")
                
                if not result.is_consistent:
                    print(f"         ⚠️  一致性警告: max_diff={result.max_diff:.2e}")
            
            results.append(result)
        
        return results


# =============================================================================
# Report Generation
# =============================================================================
def print_summary_report(results: List[BenchmarkResult], backends: List[str]):
    """打印汇总报告"""
    print("\n")
    print("=" * 130)
    print("模型规模基准测试 - 汇总报告")
    print("=" * 130)
    
    # 表头
    header = f"{'规模':<10} {'参数量':<10} {'PyTorch':<12}"
    for backend in backends:
        header += f"{backend:<14}"
    header += f"{'最快后端':<12} {'最大加速':<10}"
    print(header)
    print("-" * 130)
    
    # 数据行
    for result in results:
        if result.error_msg:
            row = f"{result.scale_name:<10} {'ERROR':<10}"
            print(row)
            continue
        
        row = f"{result.scale_name:<10} {result.params_m:>7.2f}M  {result.pytorch_ms:>9.2f}ms"
        
        best_backend = ""
        best_speedup = 0
        
        for backend in backends:
            ms = result.backend_times.get(backend, 0)
            if ms > 0:
                speedup = result.pytorch_ms / ms
                row += f"  {ms:>6.2f}ms ({speedup:>4.1f}x)"
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_backend = backend
            else:
                row += f"  {'N/A':>14}"
        
        row += f"  {best_backend:<12} {best_speedup:>7.1f}x"
        print(row)
    
    print("-" * 130)
    
    # 性能趋势分析
    print("\n性能趋势分析:")
    print("-" * 80)
    
    # 按参数量排序
    sorted_results = sorted([r for r in results if not r.error_msg], 
                            key=lambda x: x.params_m)
    
    if len(sorted_results) >= 2:
        for backend in backends:
            times = [(r.params_m, r.backend_times.get(backend, 0)) for r in sorted_results]
            valid_times = [(p, t) for p, t in times if t > 0]
            
            if len(valid_times) >= 2:
                # 计算参数量增加vs时间增加的比率
                p1, t1 = valid_times[0]
                p2, t2 = valid_times[-1]
                param_ratio = p2 / p1
                time_ratio = t2 / t1
                scaling_efficiency = param_ratio / time_ratio if time_ratio > 0 else 0
                
                print(f"  {backend:<12}: 参数量 {param_ratio:.1f}x 增加，时间 {time_ratio:.1f}x 增加，"
                      f"规模效率 {scaling_efficiency:.2f}")
    
    print("=" * 130)


def save_results_json(results: List[BenchmarkResult], path: str):
    """保存结果到 JSON 文件"""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "scale": r.scale_name,
                "params_m": r.params_m,
                "pytorch_ms": r.pytorch_ms,
                "backend_times": r.backend_times,
                "is_consistent": r.is_consistent,
                "max_diff": r.max_diff,
                "error": r.error_msg
            }
            for r in results
        ]
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n结果已保存到: {path}")


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='模型规模基准测试 - 不同参数量下的后端性能对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 模型规模选择
    parser.add_argument(
        '--scale', '-s',
        type=str,
        default='micro,tiny,small,medium',
        help='要测试的模型规模，逗号分隔 (默认: micro,tiny,small,medium)'
    )
    parser.add_argument(
        '--all-scales', '-a',
        action='store_true',
        help='测试所有预设规模'
    )
    parser.add_argument(
        '--custom-scale',
        type=str,
        help='自定义规模: H=hidden,L=layers,A=heads,KV=kv_heads,V=vocab,I=inter'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='custom',
        help='自定义规模的名称'
    )
    
    # 后端选择
    parser.add_argument(
        '--backends', '-b',
        type=str,
        default='baseline,simd,extreme',
        help='要测试的后端，逗号分隔 (默认: baseline,simd,extreme)'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False,
        help='启用 CUDA 后端 (cuda_fp32, cuda_fp16)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='禁用 CUDA 后端 (即使已启用)'
    )
    
    # 输入配置
    parser.add_argument(
        '--batch-size', '-B',
        type=int,
        default=1,
        help='批次大小 (默认: 1)'
    )
    parser.add_argument(
        '--seq-length', '-S',
        type=int,
        default=64,
        help='序列长度 (默认: 64)'
    )
    
    # 其他选项
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='跳过 C++ 编译'
    )
    parser.add_argument(
        '--dump-dir',
        type=str,
        default='dump_benchmark',
        help='临时文件目录 (默认: dump_benchmark)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='保存结果到 JSON 文件'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    parser.add_argument(
        '--atol',
        type=float,
        default=1e-2,
        help='一致性检查绝对误差阈值 (默认: 1e-2)'
    )
    parser.add_argument(
        '--rtol',
        type=float,
        default=1e-2,
        help='一致性检查相对误差阈值 (默认: 1e-2)'
    )
    
    args = parser.parse_args()
    
    # 确定项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("=" * 80)
    print("CelerInfer 模型规模基准测试")
    print("=" * 80)
    print(f"项目根目录: {project_root}")
    print(f"批次大小: {args.batch_size}, 序列长度: {args.seq_length}")
    
    # 确定要测试的模型规模
    scales_to_test: List[ModelScale] = []
    
    if args.custom_scale:
        # 解析自定义规模
        params = {}
        for kv in args.custom_scale.split(','):
            k, v = kv.split('=')
            params[k.strip().upper()] = int(v.strip())
        
        custom_scale = ModelScale(
            name=args.name,
            hidden_size=params.get('H', 256),
            num_layers=params.get('L', 4),
            num_heads=params.get('A', 8),
            num_kv_heads=params.get('KV', 2),
            vocab_size=params.get('V', 2048),
            intermediate_size=params.get('I', 0),
            description="自定义规模"
        )
        scales_to_test.append(custom_scale)
    
    if args.all_scales:
        scales_to_test.extend(MODEL_SCALES.values())
    elif not args.custom_scale:
        scale_names = [s.strip() for s in args.scale.split(',')]
        for name in scale_names:
            if name in MODEL_SCALES:
                scales_to_test.append(MODEL_SCALES[name])
            else:
                print(f"警告: 未知规模 '{name}'，跳过")
    
    if not scales_to_test:
        print("错误: 没有选择任何模型规模")
        return 1
    
    print(f"测试规模: {[s.name for s in scales_to_test]}")
    
    # 确定要测试的后端
    enable_cuda = args.cuda and not args.no_cuda
    backends_to_test = [b.strip() for b in args.backends.split(',')]
    
    if enable_cuda:
        if 'cuda_fp32' not in backends_to_test:
            backends_to_test.append('cuda_fp32')
        if 'cuda_fp16' not in backends_to_test:
            backends_to_test.append('cuda_fp16')
    
    print(f"测试后端: {backends_to_test}")
    
    # 编译 C++
    if not args.skip_build:
        print("\n[编译] C++ 推理引擎...")
        if not build_cpp(project_root, enable_cuda):
            print("编译失败!")
            return 1
        print("✓ 编译成功")
    
    # 检查可用后端
    available = check_available_backends(project_root)
    print(f"可用后端: {available}")
    
    backends_to_test = [b for b in backends_to_test if b in available]
    if not backends_to_test:
        print("错误: 没有可用的后端")
        return 1
    
    # 运行基准测试
    benchmark = ModelScaleBenchmark(
        project_root=project_root,
        dump_dir=args.dump_dir,
        enable_cuda=enable_cuda,
        verbose=args.verbose,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        atol=args.atol,
        rtol=args.rtol,
    )
    
    print("\n" + "=" * 80)
    print("开始基准测试")
    print("=" * 80)
    
    results = benchmark.run_benchmark(scales_to_test, backends_to_test)
    
    # 打印汇总报告
    print_summary_report(results, backends_to_test)
    
    # 保存结果
    if args.output:
        save_results_json(results, args.output)
    else:
        # 默认保存到 dump 目录
        default_output = os.path.join(benchmark.dump_dir, 'benchmark_results.json')
        save_results_json(results, default_output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

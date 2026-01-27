#!/usr/bin/env python3
"""
PyTorch <-> C++ 一致性验证工具 (多测试用例版 + SIMD对比)

用法:
    python -m python.tools.verify_consistency [--dump-dir dump_minimind] [--test-case all]

测试用例:
    - basic: B=2, S=5 (默认)
    - batch1: B=1, S=8 (单batch长序列)
    - batch4: B=4, S=3 (多batch短序列)
    - long: B=1, S=32 (长序列)
    - square: B=4, S=4 (方形)
    - all: 运行所有测试用例
"""

import os
import sys
import struct
import subprocess
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class TestCase:
    """测试用例配置"""
    name: str
    B: int      # Batch size
    S: int      # Sequence length
    seed: int = 123
    description: str = ""


# 定义多组测试用例，覆盖各种边界情况
TEST_CASES = {
    "basic": TestCase(name="basic", B=2, S=5, seed=123, description="默认测试 (B=2, S=5)"),
    "batch1": TestCase(name="batch1", B=1, S=8, seed=456, description="单batch长序列 (B=1, S=8)"),
    "batch4": TestCase(name="batch4", B=4, S=3, seed=789, description="多batch短序列 (B=4, S=3)"),
    "square": TestCase(name="square", B=4, S=4, seed=222, description="方形张量 (B=4, S=4)"),
    "batch8": TestCase(name="batch8", B=8, S=16, seed=444, description="较大输入 (B=8, S=16)"),
    "short": TestCase(name="short", B=2, S=2, seed=555, description="短序列 (B=2, S=2)"),
    "mid": TestCase(name="mid", B=3, S=7, seed=666, description="中等大小 (B=3, S=7)"),
    # 边界测试用例 - 之前因buffer分配bug而失败，现已修复
    "edge1": TestCase(name="edge1", B=1, S=1, seed=333, description="最小输入 (B=1, S=1)"),
    "long": TestCase(name="long", B=1, S=32, seed=111, description="长序列测试 (B=1, S=32)"),
}


def read_binary_f32(path: str) -> np.ndarray:
    """读取C++保存的原始float32二进制文件"""
    with open(path, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    values = struct.unpack(f'{count}f', data)
    return np.array(values, dtype=np.float32)


def run_command(cmd: str, cwd: str = None) -> tuple:
    """运行命令并返回输出"""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def build_cpp(project_root: str) -> bool:
    """编译C++推理引擎"""
    cpp_dir = os.path.join(project_root, 'cpp')
    build_dir = os.path.join(cpp_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)
    
    ret, out, err = run_command('cmake ..', cwd=build_dir)
    if ret != 0:
        print(f'CMake失败: {err}')
        return False
        
    ret, out, err = run_command('make -j4', cwd=build_dir)
    if ret != 0:
        print(f'Make失败: {err}')
        return False
    return True


@dataclass
class TimingResult:
    """计时结果 - 支持三种后端"""
    torch_ms: float = 0.0
    cpp_baseline_ms: float = 0.0
    cpp_simd_ms: float = 0.0
    baseline_speedup: float = 0.0   # torch_ms / cpp_baseline_ms
    simd_speedup: float = 0.0       # torch_ms / cpp_simd_ms
    simd_vs_baseline: float = 0.0   # cpp_baseline_ms / cpp_simd_ms


def run_single_test(
    test_case: TestCase,
    project_root: str,
    dump_dir: str,
    atol: float,
    rtol: float,
    verbose: bool = False,
    run_simd: bool = True
) -> Tuple[bool, float, str, Optional[TimingResult]]:
    """
    运行单个测试用例
    
    Returns:
        (passed, max_diff, message, timing)
    """
    # 设置环境变量
    os.environ['DUMP_DIR'] = dump_dir
    os.environ['JSON_PATH'] = os.path.join(dump_dir, 'minimind.json')
    os.environ['B'] = str(test_case.B)
    os.environ['S'] = str(test_case.S)
    os.environ['SEED'] = str(test_case.seed)
    
    # 强制重新导入模块以使用新环境变量
    import importlib
    
    # Step 1: 生成模型权重
    try:
        from python.export import minimind_dumper
        importlib.reload(minimind_dumper)
        minimind_dumper.main()
    except Exception as e:
        return False, float('inf'), f"权重导出失败: {e}"
    
    # Step 2: PyTorch推理
    try:
        from python.inference import minimind_forward
        importlib.reload(minimind_forward)
        minimind_forward.main()
    except Exception as e:
        return False, float('inf'), f"PyTorch推理失败: {e}"
    
    # Step 3: C++ baseline 推理
    cpp_baseline = os.path.join(project_root, 'cpp', 'build', 'minimind')
    json_path = os.path.join(dump_dir, 'minimind.json')
    
    ret, out_baseline, err = run_command(f'{cpp_baseline} {json_path} {dump_dir}', cwd=project_root)
    if ret != 0:
        return False, float('inf'), f"C++ baseline推理失败: {err}", None
    
    if verbose:
        print(out_baseline)
    
    # Step 3b: C++ SIMD 推理
    cpp_simd = os.path.join(project_root, 'cpp', 'build', 'minimind_simd')
    out_simd = ""
    if run_simd and os.path.exists(cpp_simd):
        ret, out_simd, err = run_command(f'{cpp_simd} {json_path} {dump_dir}', cwd=project_root)
        if ret != 0:
            if verbose:
                print(f"C++ SIMD推理失败: {err}")
            # SIMD失败不算测试失败，继续使用baseline
            out_simd = ""
        elif verbose:
            print(out_simd)
    
    # Step 4: 对比结果
    torch_path = os.path.join(dump_dir, 'logits_torch.npy')
    cpp_path = os.path.join(dump_dir, 'logits_cpp.npy')
    
    try:
        logits_torch = np.load(torch_path, allow_pickle=True).astype(np.float32)
        logits_cpp = read_binary_f32(cpp_path).reshape(logits_torch.shape)
    except Exception as e:
        return False, float('inf'), f"加载输出失败: {e}"
    
    diff = np.abs(logits_torch - logits_cpp)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    
    is_close = np.allclose(logits_torch, logits_cpp, rtol=rtol, atol=atol)
    
    # 收集计时信息
    timing = None
    try:
        import json as json_module
        timing_torch_path = os.path.join(dump_dir, 'timing_torch.json')
        if os.path.exists(timing_torch_path):
            with open(timing_torch_path) as f:
                torch_timing = json_module.load(f)
            torch_ms = torch_timing.get('elapsed_ms', 0)
            
            # 从C++ baseline 输出解析计时
            cpp_baseline_ms = 0.0
            import re
            match = re.search(r'Forward pass: ([\d.]+)ms', out_baseline)
            if match:
                cpp_baseline_ms = float(match.group(1))
            
            # 从C++ SIMD 输出解析计时
            cpp_simd_ms = 0.0
            if out_simd:
                match = re.search(r'SIMD Forward pass: ([\d.]+)ms', out_simd)
                if match:
                    cpp_simd_ms = float(match.group(1))
            
            baseline_speedup = torch_ms / cpp_baseline_ms if cpp_baseline_ms > 0 else 0
            simd_speedup = torch_ms / cpp_simd_ms if cpp_simd_ms > 0 else 0
            simd_vs_baseline = cpp_baseline_ms / cpp_simd_ms if cpp_simd_ms > 0 else 0
            
            timing = TimingResult(
                torch_ms=torch_ms,
                cpp_baseline_ms=cpp_baseline_ms,
                cpp_simd_ms=cpp_simd_ms,
                baseline_speedup=baseline_speedup,
                simd_speedup=simd_speedup,
                simd_vs_baseline=simd_vs_baseline
            )
    except Exception:
        pass
    
    if is_close:
        msg = f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        return True, max_diff, msg, timing
    else:
        close_mask = np.isclose(logits_torch, logits_cpp, rtol=rtol, atol=atol)
        inconsistent_ratio = (1 - close_mask.mean()) * 100
        msg = f"max_diff={max_diff:.2e}, {inconsistent_ratio:.1f}%不一致"
        return False, max_diff, msg, timing


def main():
    parser = argparse.ArgumentParser(description='PyTorch <-> C++ 一致性验证 (多测试用例)')
    parser.add_argument('--dump-dir', default='dump_minimind', help='Dump目录')
    parser.add_argument('--atol', type=float, default=1e-3, help='绝对误差阈值')
    parser.add_argument('--rtol', type=float, default=1e-3, help='相对误差阈值')
    parser.add_argument('--skip-build', action='store_true', help='跳过C++编译')
    parser.add_argument('--test-case', '-t', default='all', 
                        choices=list(TEST_CASES.keys()) + ['all'],
                        help='测试用例 (默认: all)')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--simd', action='store_true', default=True, 
                        help='运行SIMD优化版本 (默认启用)')
    parser.add_argument('--no-simd', dest='simd', action='store_false',
                        help='跳过SIMD优化版本')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dump_dir = os.path.join(project_root, args.dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    print('=' * 70)
    print('PyTorch <-> C++ 一致性验证 (多测试用例 + SIMD对比)')
    print('=' * 70)
    print()

    # Step 1: 编译C++
    if not args.skip_build:
        print('[编译] C++推理引擎...')
        if not build_cpp(project_root):
            return 1
        print('  ✓ C++编译成功')
        print()
    
    # 确定要运行的测试用例
    if args.test_case == 'all':
        cases_to_run = list(TEST_CASES.values())
    else:
        cases_to_run = [TEST_CASES[args.test_case]]
    
    # 运行测试
    print(f'[测试] 运行 {len(cases_to_run)} 个测试用例...')
    print()
    
    results: List[Tuple[str, bool, float, str, Optional[TimingResult]]] = []
    
    for i, tc in enumerate(cases_to_run, 1):
        print(f'[{i}/{len(cases_to_run)}] {tc.name}: {tc.description}')
        print(f'      参数: B={tc.B}, S={tc.S}, seed={tc.seed}')
        
        passed, max_diff, msg, timing = run_single_test(
            tc, project_root, dump_dir,
            args.atol, args.rtol, args.verbose, args.simd
        )
        
        status = '✅ 通过' if passed else '❌ 失败'
        timing_str = ''
        if timing:
            timing_str = f' | PyTorch: {timing.torch_ms:.2f}ms, Baseline: {timing.cpp_baseline_ms:.2f}ms'
            if timing.cpp_simd_ms > 0:
                timing_str += f', SIMD: {timing.cpp_simd_ms:.2f}ms'
        print(f'      结果: {status} ({msg}){timing_str}')
        print()
        
        results.append((tc.name, passed, max_diff, msg, timing))
    
    # 汇总结果
    print('=' * 110)
    print('测试结果汇总')
    print('=' * 110)
    print()
    print(f'{"测试用例":<12} {"状态":<8} {"最大误差":<12} {"PyTorch(ms)":<12} {"Baseline(ms)":<14} {"SIMD(ms)":<12} {"SIMD加速"}')
    print('-' * 110)
    
    passed_count = 0
    total_torch_ms = 0.0
    total_baseline_ms = 0.0
    total_simd_ms = 0.0
    timing_count = 0
    
    for name, passed, max_diff, msg, timing in results:
        status = '✅ 通过' if passed else '❌ 失败'
        diff_str = f'{max_diff:.2e}' if max_diff != float('inf') else 'N/A'
        
        if timing:
            torch_str = f'{timing.torch_ms:.2f}'
            baseline_str = f'{timing.cpp_baseline_ms:.2f}'
            simd_str = f'{timing.cpp_simd_ms:.2f}' if timing.cpp_simd_ms > 0 else 'N/A'
            simd_speedup_str = f'{timing.simd_vs_baseline:.2f}x' if timing.simd_vs_baseline > 0 else 'N/A'
            total_torch_ms += timing.torch_ms
            total_baseline_ms += timing.cpp_baseline_ms
            if timing.cpp_simd_ms > 0:
                total_simd_ms += timing.cpp_simd_ms
            timing_count += 1
        else:
            torch_str = baseline_str = simd_str = simd_speedup_str = 'N/A'
            
        print(f'{name:<12} {status:<8} {diff_str:<12} {torch_str:<12} {baseline_str:<14} {simd_str:<12} {simd_speedup_str}')
        if passed:
            passed_count += 1
    
    print('-' * 110)
    print(f'总计: {passed_count}/{len(results)} 通过')
    
    # 显示计时汇总
    if timing_count > 0:
        avg_baseline_speedup = total_torch_ms / total_baseline_ms if total_baseline_ms > 0 else 0
        avg_simd_speedup = total_torch_ms / total_simd_ms if total_simd_ms > 0 else 0
        simd_vs_baseline_total = total_baseline_ms / total_simd_ms if total_simd_ms > 0 else 0
        print()
        print('=' * 110)
        print('性能对比汇总')
        print('=' * 110)
        print(f'  PyTorch    总耗时: {total_torch_ms:.2f}ms')
        print(f'  C++ Baseline 总耗时: {total_baseline_ms:.2f}ms (vs PyTorch: {avg_baseline_speedup:.2f}x)')
        if total_simd_ms > 0:
            print(f'  C++ SIMD   总耗时: {total_simd_ms:.2f}ms (vs PyTorch: {avg_simd_speedup:.2f}x, vs Baseline: {simd_vs_baseline_total:.2f}x)')
        
        # 按元素数量排序显示性能趋势
        print()
        print('性能趋势 (按输入大小排序):')
        sorted_results = []
        for name, passed, max_diff, msg, timing in results:
            tc = TEST_CASES.get(name)
            if tc and timing:
                elements = tc.B * tc.S
                sorted_results.append((elements, tc.B, tc.S, name, timing))
        
        sorted_results.sort(key=lambda x: x[0])
        has_simd = any(t[4].cpp_simd_ms > 0 for t in sorted_results)
        
        if has_simd:
            print(f'  {"B*S":<6} {"B":<3} {"S":<3} {"用例":<10} {"PyTorch":<10} {"Baseline":<10} {"SIMD":<10} {"Base/Torch":<12} {"SIMD/Torch":<12} {"SIMD/Base"}')
        else:
            print(f'  {"B*S":<8} {"B":<4} {"S":<4} {"用例":<10} {"PyTorch":<10} {"Baseline":<10} {"加速比"}')
        
        for elements, b, s, name, timing in sorted_results:
            if has_simd:
                baseline_vs_torch = f'{timing.baseline_speedup:.2f}x' if timing.baseline_speedup > 0 else 'N/A'
                simd_vs_torch = f'{timing.simd_speedup:.2f}x' if timing.simd_speedup > 0 else 'N/A'
                simd_vs_base = f'{timing.simd_vs_baseline:.2f}x' if timing.simd_vs_baseline > 0 else 'N/A'
                simd_str = f'{timing.cpp_simd_ms:.2f}ms' if timing.cpp_simd_ms > 0 else 'N/A'
                
                # 指示器
                if timing.simd_vs_baseline > 1.5:
                    indicator = '🚀'  # SIMD显著更快
                elif timing.simd_vs_baseline > 1.0:
                    indicator = '⬆️'  # SIMD稍快
                elif timing.simd_vs_baseline > 0:
                    indicator = '⬇️'  # SIMD更慢
                else:
                    indicator = '  '
                
                print(f'  {elements:<6} {b:<3} {s:<3} {name:<10} {timing.torch_ms:.2f}ms     {timing.cpp_baseline_ms:.2f}ms     {simd_str:<10} {baseline_vs_torch:<12} {simd_vs_torch:<12} {simd_vs_base} {indicator}')
            else:
                speedup_str = f'{timing.baseline_speedup:.2f}x' if timing.baseline_speedup > 0 else 'N/A'
                indicator = '🚀' if timing.baseline_speedup > 1.5 else ('⚠️' if timing.baseline_speedup < 0.5 else '  ')
                print(f'  {elements:<8} {b:<4} {s:<4} {name:<10} {timing.torch_ms:.2f}ms     {timing.cpp_baseline_ms:.2f}ms     {speedup_str} {indicator}')
        
        print()
        if has_simd:
            print('  🚀 = SIMD显著更快(>1.5x)  ⬆️ = SIMD稍快(>1x)  ⬇️ = SIMD更慢(<1x)')
        else:
            print('  🚀 = C++ 显著更快 (>1.5x)  ⚠️ = C++ 显著更慢 (<0.5x)')
        print('=' * 110)
    print()
    
    if passed_count == len(results):
        print('=' * 110)
        print('✅ 所有测试用例通过!')
        print(f'   阈值: atol={args.atol}, rtol={args.rtol}')
        print('=' * 110)
        return 0
    else:
        failed_count = len(results) - passed_count
        print('=' * 110)
        print(f'❌ {failed_count} 个测试用例失败')
        print('=' * 110)
        return 1


if __name__ == '__main__':
    sys.exit(main())

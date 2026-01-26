#!/usr/bin/env python3
"""
PyTorch <-> C++ 一致性验证工具 (多测试用例版)

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


def run_single_test(
    test_case: TestCase,
    project_root: str,
    dump_dir: str,
    atol: float,
    rtol: float,
    verbose: bool = False
) -> Tuple[bool, float, str]:
    """
    运行单个测试用例
    
    Returns:
        (passed, max_diff, message)
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
    
    # Step 3: C++推理
    cpp_binary = os.path.join(project_root, 'cpp', 'build', 'base_line_micro')
    json_path = os.path.join(dump_dir, 'minimind.json')
    
    ret, out, err = run_command(f'{cpp_binary} {json_path} {dump_dir}', cwd=project_root)
    if ret != 0:
        return False, float('inf'), f"C++推理失败: {err}"
    
    if verbose:
        print(out)
    
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
    
    if is_close:
        msg = f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        return True, max_diff, msg
    else:
        close_mask = np.isclose(logits_torch, logits_cpp, rtol=rtol, atol=atol)
        inconsistent_ratio = (1 - close_mask.mean()) * 100
        msg = f"max_diff={max_diff:.2e}, {inconsistent_ratio:.1f}%不一致"
        return False, max_diff, msg


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
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dump_dir = os.path.join(project_root, args.dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    print('=' * 70)
    print('PyTorch <-> C++ 一致性验证 (多测试用例)')
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
    
    results: List[Tuple[str, bool, float, str]] = []
    
    for i, tc in enumerate(cases_to_run, 1):
        print(f'[{i}/{len(cases_to_run)}] {tc.name}: {tc.description}')
        print(f'      参数: B={tc.B}, S={tc.S}, seed={tc.seed}')
        
        passed, max_diff, msg = run_single_test(
            tc, project_root, dump_dir,
            args.atol, args.rtol, args.verbose
        )
        
        status = '✅ 通过' if passed else '❌ 失败'
        print(f'      结果: {status} ({msg})')
        print()
        
        results.append((tc.name, passed, max_diff, msg))
    
    # 汇总结果
    print('=' * 70)
    print('测试结果汇总')
    print('=' * 70)
    print()
    print(f'{"测试用例":<12} {"状态":<8} {"最大误差":<15} {"详情"}')
    print('-' * 70)
    
    passed_count = 0
    for name, passed, max_diff, msg in results:
        status = '✅ 通过' if passed else '❌ 失败'
        diff_str = f'{max_diff:.2e}' if max_diff != float('inf') else 'N/A'
        print(f'{name:<12} {status:<8} {diff_str:<15} {msg}')
        if passed:
            passed_count += 1
    
    print('-' * 70)
    print(f'总计: {passed_count}/{len(results)} 通过')
    print()
    
    if passed_count == len(results):
        print('=' * 70)
        print('✅ 所有测试用例通过!')
        print(f'   阈值: atol={args.atol}, rtol={args.rtol}')
        print('=' * 70)
        return 0
    else:
        failed_count = len(results) - passed_count
        print('=' * 70)
        print(f'❌ {failed_count} 个测试用例失败')
        print('=' * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

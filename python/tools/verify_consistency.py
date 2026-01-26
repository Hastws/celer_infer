#!/usr/bin/env python3
"""
PyTorch <-> C++ 一致性验证工具

用法:
    python -m python.tools.verify_consistency [--dump-dir dump_minimind]
"""

import os
import sys
import struct
import subprocess
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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


def main():
    parser = argparse.ArgumentParser(description='PyTorch <-> C++ 一致性验证')
    parser.add_argument('--dump-dir', default='dump_minimind', help='Dump目录')
    parser.add_argument('--atol', type=float, default=1e-3, help='绝对误差阈值')
    parser.add_argument('--rtol', type=float, default=1e-3, help='相对误差阈值')
    parser.add_argument('--skip-build', action='store_true', help='跳过C++编译')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dump_dir = os.path.join(project_root, args.dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    print('=' * 60)
    print('PyTorch <-> C++ 一致性验证')
    print('=' * 60)
    print()

    # Step 1: 编译C++
    if not args.skip_build:
        print('[1/4] 编译C++推理引擎...')
        cpp_dir = os.path.join(project_root, 'cpp')
        build_dir = os.path.join(cpp_dir, 'build')
        
        os.makedirs(build_dir, exist_ok=True)
        
        ret, out, err = run_command('cmake ..', cwd=build_dir)
        if ret != 0:
            print(f'CMake失败: {err}')
            return 1
            
        ret, out, err = run_command('make -j4', cwd=build_dir)
        if ret != 0:
            print(f'Make失败: {err}')
            return 1
        print('  ✓ C++编译成功')
    else:
        print('[1/4] 跳过C++编译')

    # Step 2: 生成模型权重
    print('[2/4] 生成模型权重...')
    os.environ['DUMP_DIR'] = dump_dir
    os.environ['JSON_PATH'] = os.path.join(dump_dir, 'minimind.json')
    
    from python.export.minimind_dumper import main as dump_main
    dump_main()
    print('  ✓ 权重导出完成')

    # Step 3: PyTorch推理
    print('[3/4] 运行PyTorch推理...')
    from python.inference.minimind_forward import main as forward_main
    forward_main()
    print('  ✓ PyTorch推理完成')

    # Step 4: C++推理
    print('[4/4] 运行C++推理...')
    cpp_binary = os.path.join(project_root, 'cpp', 'build', 'base_line_micro')
    json_path = os.path.join(dump_dir, 'minimind.json')
    
    ret, out, err = run_command(f'{cpp_binary} {json_path} {dump_dir}', cwd=project_root)
    if ret != 0:
        print(f'C++推理失败: {err}')
        return 1
    print(out)
    print('  ✓ C++推理完成')

    # Step 5: 对比结果
    print()
    print('=' * 60)
    print('验证结果')
    print('=' * 60)

    torch_path = os.path.join(dump_dir, 'logits_torch.npy')
    cpp_path = os.path.join(dump_dir, 'logits_cpp.npy')

    logits_torch = np.load(torch_path, allow_pickle=True).astype(np.float32)
    logits_cpp = read_binary_f32(cpp_path).reshape(logits_torch.shape)

    print(f'PyTorch: shape={logits_torch.shape}, min={logits_torch.min():.6f}, max={logits_torch.max():.6f}')
    print(f'C++:     shape={logits_cpp.shape}, min={logits_cpp.min():.6f}, max={logits_cpp.max():.6f}')

    diff = np.abs(logits_torch - logits_cpp)
    print()
    print(f'最大绝对差异: {diff.max():.10f}')
    print(f'平均绝对差异: {diff.mean():.10f}')

    is_close = np.allclose(logits_torch, logits_cpp, rtol=args.rtol, atol=args.atol)

    print()
    if is_close:
        print('=' * 60)
        print('✅ 验证通过! PyTorch和C++输出一致!')
        print(f'   最大误差 {diff.max():.6f} < 阈值 (atol={args.atol}, rtol={args.rtol})')
        print('=' * 60)
        return 0
    else:
        close_mask = np.isclose(logits_torch, logits_cpp, rtol=args.rtol, atol=args.atol)
        inconsistent_ratio = (1 - close_mask.mean()) * 100
        print('=' * 60)
        print(f'❌ 验证失败: {inconsistent_ratio:.2f}% 元素不一致')
        print('=' * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())

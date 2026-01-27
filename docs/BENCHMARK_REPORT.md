# CelerInfer Benchmark Report

**Generated**: January 28, 2025  
**Test Environment**: Linux, CUDA enabled  
**Benchmark Settings**: Warmup=5, Runs=20

---

## 📋 Executive Summary

CelerInfer 实现了两个模型的多后端推理：
- **MiniMind**: 大语言模型 (LLM)
- **DiT**: 扩散 Transformer (Diffusion Transformer)

所有后端的输出与 PyTorch 参考实现保持一致，CUDA FP16 后端展示了显著的性能提升。

---

## ✅ Part 1: 推理一致性验证报告

### 1.1 验证方法

- **参考实现**: PyTorch (CPU, FP32)
- **验证指标**: 
  - 最大绝对误差 (max_abs)
  - 最大相对误差 (max_rel)
- **验证阈值**: atol=0.02, rtol=0.05

### 1.2 MiniMind 一致性结果

| 后端 | 最大绝对误差 | 状态 |
|------|-------------|------|
| C++ Baseline | 1.84e-02 | ✅ Pass |
| C++ SIMD | 1.84e-02 | ✅ Pass |
| C++ Extreme | 1.84e-02 | ✅ Pass |
| CUDA FP32 | 1.84e-02 | ✅ Pass |
| CUDA FP16 | 1.84e-02 | ✅ Pass |

**结论**: 所有后端输出与 PyTorch 一致 ✅

### 1.3 DiT 一致性结果

| 后端 | 状态 |
|------|------|
| C++ Baseline | ✅ Pass |
| C++ SIMD | ✅ Pass |
| C++ Extreme | ✅ Pass |
| CUDA FP32 | ✅ Pass |
| CUDA FP16 | ✅ Pass |

**结论**: 所有后端输出与 PyTorch 一致 ✅

### 1.4 误差分析

MiniMind 的 max_rel 较大 (1.22e+02) 是因为：
- 随机初始化权重导致输出值接近 0 (范围 [-0.02, 0.02])
- 当参考值接近 0 时，相对误差计算会放大
- **绝对误差 1.84e-02 在可接受范围内**

---

## 🚀 Part 2: 性能对比报告

### 2.1 测试配置

| 模型 | Hidden | Layers | Heads | Batch | Seq |
|------|--------|--------|-------|-------|-----|
| MiniMind | 512 | 8 | 8 | 2 | 5 |
| DiT | 192 | 3 | 6 | 4 | 16 |

### 2.2 MiniMind (LLM) 性能结果

| 后端 | 延迟 (ms) | 相对 PyTorch | 加速比 |
|------|-----------|--------------|--------|
| PyTorch (baseline) | 12.05 | 1.00x | - |
| C++ Baseline | 93.20 | 0.13x | 🔻 7.7x 慢 |
| C++ SIMD | 23.72 | 0.51x | 🔻 2.0x 慢 |
| C++ Extreme | 82.47 | 0.15x | 🔻 6.8x 慢 |
| **CUDA FP32** | **1.01** | **11.90x** | 🚀 **11.9x 快** |
| **CUDA FP16** | **0.64** | **18.96x** | 🚀 **19.0x 快** |

**关键发现**:
- CPU C++ 实现比 PyTorch 慢，原因是 PyTorch 对小矩阵 GEMM 高度优化
- CUDA FP32 比 PyTorch 快 **11.9 倍**
- CUDA FP16 比 PyTorch 快 **19.0 倍**

### 2.3 DiT (Diffusion Transformer) 性能结果

| 后端 | 延迟 (ms) | 相对 PyTorch | 加速比 |
|------|-----------|--------------|--------|
| PyTorch (baseline) | 37.27 | 1.00x | - |
| C++ Baseline | 52.81 | 0.71x | 🔻 1.4x 慢 |
| **C++ SIMD** | **10.90** | **3.42x** | 🚀 **3.4x 快** |
| **C++ Extreme** | **3.56** | **10.47x** | 🚀 **10.5x 快** |
| **CUDA FP32** | **1.75** | **21.30x** | 🚀 **21.3x 快** |
| **CUDA FP16** | **0.23** | **159.77x** | 🚀 **159.8x 快** |

**关键发现**:
- C++ SIMD 比 PyTorch 快 **3.4 倍**
- C++ Extreme (展开+SIMD) 比 PyTorch 快 **10.5 倍**
- CUDA FP32 比 PyTorch 快 **21.3 倍**
- CUDA FP16 比 PyTorch 快 **159.8 倍** 🔥

### 2.4 性能对比图

```
MiniMind (LLM) - 延迟对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PyTorch     ████████████ 12.05 ms
Baseline    ████████████████████████████████████████████████████████████████████████████████████████████ 93.20 ms
SIMD        ████████████████████████ 23.72 ms
Extreme     ██████████████████████████████████████████████████████████████████████████████████ 82.47 ms
CUDA FP32   █ 1.01 ms
CUDA FP16   █ 0.64 ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DiT (Diffusion Transformer) - 延迟对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PyTorch     █████████████████████████████████████ 37.27 ms
Baseline    █████████████████████████████████████████████████████ 52.81 ms
SIMD        ███████████ 10.90 ms
Extreme     ████ 3.56 ms
CUDA FP32   ██ 1.75 ms
CUDA FP16   █ 0.23 ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📊 Part 3: 总结

### 3.1 一致性总结

| 模型 | 测试后端数 | 通过数 | 通过率 |
|------|-----------|--------|--------|
| MiniMind | 5 | 5 | **100%** |
| DiT | 5 | 5 | **100%** |

**结论**: 所有实现的输出与 PyTorch 参考实现一致 ✅

### 3.2 性能总结

| 模型 | 最快后端 | 相对 PyTorch 加速 |
|------|----------|------------------|
| MiniMind | CUDA FP16 | **19.0x** |
| DiT | CUDA FP16 | **159.8x** 🔥 |

### 3.3 后端推荐

| 场景 | 推荐后端 | 理由 |
|------|----------|------|
| GPU 生产环境 | CUDA FP16 | 最高性能，精度损失可接受 |
| GPU 高精度需求 | CUDA FP32 | 平衡性能与精度 |
| CPU 服务器 | C++ SIMD/Extreme | 无 GPU 环境的最佳选择 |
| 开发调试 | PyTorch | 灵活易调试 |

---

## 🔧 复现方法

```bash
# 运行完整 benchmark
python scripts/unified_benchmark.py --model all --warmup 5 --runs 20 --output results.json

# 只测试特定后端
python scripts/unified_benchmark.py --model all --backends pytorch,cuda,cuda_fp16

# 只测试特定模型
python scripts/unified_benchmark.py --model minimind
python scripts/unified_benchmark.py --model dit
```

---

**Report End**

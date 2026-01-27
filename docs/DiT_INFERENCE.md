# DiT (Diffusion Transformer) 推理流程文档

## 概述

DiT 是用于自动驾驶轨迹规划的扩散模型，使用 Transformer 架构进行去噪和预测。

## 项目结构

```
CelerInfer/
├── python/
│   ├── core/
│   │   └── dit.py              # DiT 模型定义
│   ├── export/
│   │   └── dit_dumper.py       # 权重导出器 (PyTorch → JSON)
│   ├── inference/
│   │   └── dit_forward.py      # PyTorch 推理
│   ├── validate/
│   │   └── dit_validate.py     # 一致性验证
│   └── tools/
│       └── dit_benchmark.py    # 多平台基准测试
├── cpp/
│   └── src/models/
│       ├── dit.cpp             # C++ 基线实现
│       ├── dit_simd.cpp        # AVX2 SIMD 优化
│       └── dit_extreme.cpp     # OpenMP + AVX2 极限优化
└── models/
    └── dit/                    # 模型配置和权重
```

## 模型架构

### Encoder (编码器)
- **Position Embedding**: Linear(7 → hidden_dim)
- **Neighbor Encoder**: AgentFusionEncoder (MLP-Mixer blocks)
- **Static Encoder**: StaticFusionEncoder (MLP)
- **Lane Encoder**: LaneFusionEncoder (MLP-Mixer blocks)
- **Fusion Encoder**: FusionEncoder (Self-Attention blocks)

### Decoder (解码器)
- **Route Encoder**: RouteEncoder (MLP-Mixer)
- **Agent Embedding**: Embedding(2 → hidden_dim)
- **Preproj**: MLP
- **Timestep Embedder**: TimestepEmbedder (MLP + sinusoidal)
- **DiT Blocks**: DiTBlock with adaLN-Zero conditioning
- **Final Layer**: FinalLayer with projection MLP

## 使用方法

### 1. 验证 Python 实现
```bash
python -m python.validate.dit_validate
```

### 2. 导出模型权重
```bash
# 使用默认配置
python -m python.export.dit_dumper

# 自定义配置
HIDDEN=256 DEPTH=4 HEADS=8 python -m python.export.dit_dumper
```

### 3. 构建 C++ 推理引擎
```bash
cd cpp
mkdir -p build && cd build
cmake -DENABLE_SIMD=ON -DENABLE_OPENMP=ON ..
make dit dit_simd dit_extreme -j4
```

### 4. 运行基准测试
```bash
# Python 基准测试
python -m python.tools.dit_benchmark

# 指定后端和规模
python -m python.tools.dit_benchmark --backends pytorch baseline simd extreme --scales tiny small base

# 保存结果
python -m python.tools.dit_benchmark --output benchmark_results.json
```

## 预定义模型规模

| Scale  | hidden_dim | encoder_depth | decoder_depth | num_heads | ~Params |
|--------|------------|---------------|---------------|-----------|---------|
| tiny   | 96         | 2             | 2             | 4         | ~1M     |
| small  | 128        | 3             | 4             | 4         | ~3M     |
| base   | 192        | 4             | 6             | 6         | ~8M     |
| medium | 256        | 6             | 8             | 8         | ~20M    |
| large  | 384        | 8             | 10            | 8         | ~50M    |
| xlarge | 512        | 10            | 12            | 8         | ~100M   |

## C++ 后端说明

### baseline (dit)
- 纯 C++ 实现
- 无 SIMD 优化
- 适合调试和验证

### simd (dit_simd)
- AVX2/FMA 向量化
- 单线程
- 比 baseline 快 2-4x

### extreme (dit_extreme)
- OpenMP 多线程
- AVX2/FMA 向量化
- 缓存友好的分块
- 比 baseline 快 4-8x

## 输入格式

| 字段名 | 形状 | 描述 |
|--------|------|------|
| ego_current_state | (B, 10) | 自车当前状态 |
| neighbor_agents_past | (B, agent_num, time_len, state_dim) | 邻车历史轨迹 |
| static_objects | (B, static_num, static_dim) | 静态障碍物 |
| lanes | (B, lane_num, lane_len, lane_dim) | 车道线 |
| lanes_speed_limit | (B, lane_num, 1) | 车道限速 |
| lanes_has_speed_limit | (B, lane_num, 1) | 是否有限速 |
| route_lanes | (B, route_num, route_len, route_dim) | 路由车道 |
| sampled_trajectories | (B, P, future_len+1, 4) | 采样轨迹 |
| diffusion_time | (B,) | 扩散时间步 |

## 输出格式

| 字段名 | 形状 | 描述 |
|--------|------|------|
| encoding | (B, token_num, hidden_dim) | 编码器输出 |
| prediction | (B, P, future_len, 4) | 预测轨迹 |

## 注意事项

1. **设备兼容性**: C++ 实现目前只支持 CPU，CUDA 版本待实现
2. **精度**: 使用 FP32 精度，FP16 版本待优化
3. **内存**: 大模型需要较多内存，注意配置合适的批量大小
4. **扩散采样**: C++ 实现目前只包含单步去噪，完整采样循环需要额外实现

## 性能预期

| Scale | PyTorch (CPU) | C++ Extreme | 加速比 |
|-------|---------------|-------------|--------|
| tiny  | ~10ms         | ~3ms        | ~3x    |
| small | ~25ms         | ~7ms        | ~3.5x  |
| base  | ~50ms         | ~12ms       | ~4x    |
| medium| ~100ms        | ~25ms       | ~4x    |

*实际性能取决于硬件配置和具体输入大小*

# CelerInfer - High-Performance Multi-Model Inference Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)
![C++](https://img.shields.io/badge/C%2B%2B-14-blue)
![CUDA](https://img.shields.io/badge/CUDA-FP16%2FFP32-green)

A hybrid **C++/Python inference framework** with multiple optimized backends, achieving up to **160x speedup** over PyTorch.

---

## 🚀 Performance Highlights

<table>
<tr>
<td width="50%">

### MiniMind (LLM)
*Hidden=512, Layers=8, Heads=8*

| Backend | Latency | Speedup |
|---------|---------|---------|
| PyTorch | 12.05 ms | 1.0x |
| C++ SIMD | 23.72 ms | 0.5x |
| **CUDA FP32** | **1.01 ms** | **11.9x** |
| **CUDA FP16** | **0.64 ms** | **18.9x** 🚀 |

</td>
<td width="50%">

### DiT (Diffusion Transformer)
*Hidden=192, Layers=3, Heads=6*

| Backend | Latency | Speedup |
|---------|---------|---------|
| PyTorch | 37.27 ms | 1.0x |
| **C++ SIMD** | **10.90 ms** | **3.4x** |
| **C++ Extreme** | **3.56 ms** | **10.5x** |
| **CUDA FP32** | **1.75 ms** | **21.3x** |
| **CUDA FP16** | **0.23 ms** | **159.8x** 🔥 |

</td>
</tr>
</table>

> ✅ **100% Consistency**: All backends verified against PyTorch reference (max error < 0.02)

---

## ✨ Key Features

- 🔄 **Multi-Model**: LLM (MiniMind) + Diffusion Transformer (DiT)
- ⚡ **6 Backends**: PyTorch, C++ Baseline, SIMD, Extreme, CUDA FP32, CUDA FP16
- ✅ **Verified**: Automatic consistency checking across all implementations
- 📊 **Benchmarked**: Unified benchmark tool with detailed reports

---

## 🏃 Quick Start

```bash
# 1. Setup environment
conda create -n CelerInfer python=3.12 && conda activate CelerInfer
pip install torch numpy transformers

# 2. Build C++ backends
bash scripts/build_cpp.sh

# 3. Run benchmark
python scripts/unified_benchmark.py --model all --warmup 5 --runs 20

# 4. Verify consistency
python -m python.tools.verify_consistency
```

---

## 📁 Project Structure

```
CelerInfer/
├── python/                 # Python models & tools
│   ├── core/              # Model definitions (MiniMind, DiT)
│   ├── export/            # Weight export (PyTorch → JSON)
│   └── tools/             # Verification & utilities
├── cpp/                    # C++ inference engine
│   └── src/models/        # minimind.cpp, dit.cpp, *_simd, *_cuda, etc.
├── scripts/               # Build & benchmark scripts
│   └── unified_benchmark.py
└── docs/                  # Documentation
    └── BENCHMARK_REPORT.md
```

---

## 🎯 Supported Models

| Model | Type | Backends | Best Speedup |
|-------|------|----------|--------------|
| **MiniMind** | LLM (Decoder-only) | All 6 | 18.9x (CUDA FP16) |
| **DiT** | Diffusion Transformer | All 6 | **159.8x** (CUDA FP16) 🔥 |

---

## 📊 Run Benchmarks

```bash
# Full benchmark (all models, all backends)
python scripts/unified_benchmark.py --model all --warmup 5 --runs 20 --output results.json

# Specific model
python scripts/unified_benchmark.py --model minimind
python scripts/unified_benchmark.py --model dit

# CPU only (for CI without GPU)
python scripts/unified_benchmark.py --backends pytorch,baseline,simd
```

See [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) for detailed analysis.

---

## 🔧 Build & Test

```bash
# Build all backends
bash scripts/build_cpp.sh

# Build specific targets
cd cpp/build && make minimind_cuda_fp16 dit_cuda_fp16

# Run validation
python -m python.tools.verify_consistency --atol 0.02 --rtol 0.05
```

---

## 📝 Environment

| Component | Version |
|-----------|---------|
| Python | 3.12+ |
| C++ | C++14 |
| CUDA | 11.0+ (optional) |
| CMake | 3.16+ |

**Dependencies**: torch, numpy, transformers, nlohmann/json (included)

---

## 📖 Documentation

- [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) - Performance analysis
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Design & extension guide
- [docs/MODELS.md](docs/MODELS.md) - Model specifications

---

## 📜 License

MIT License - See [LICENSE](LICENSE)

---

**Version**: 0.2.0 | **Updated**: January 28, 2025 | **Status**: ✅ Production Ready

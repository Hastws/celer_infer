# CelerInfer - Modular Multi-Model LLM Inference Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12.11-blue)
![C++](https://img.shields.io/badge/C%2B%2B-14-blue)

CelerInfer is a hybrid **C++/Python LLM inference framework** designed for efficient model inference and debugging. It bridges PyTorch model training with optimized C++ tensor operations, supporting multiple model architectures (MiniMind, LLAMA, etc.).

## ✨ Key Features

- 🔄 **Multi-Model Support** - Modular architecture for easy model integration
- 🚀 **Unified CLI** - Single command interface for all operations (`python -m python <cmd>`)
- 📊 **PyTorch ↔ C++ Verification** - Automatic consistency checking
- 🔍 **Layer-wise Debugging** - Extract and compare intermediate outputs
- 📚 **Complete Documentation** - Architecture guide and extension examples
- ⚙️ **Automated Scripts** - Build, validate, and cleanup with convenience scripts

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n CelerInfer python=3.12.11
conda activate CelerInfer

# Install dependencies (if needed)
pip install torch transformers numpy
```

### 2. Using the Unified CLI

```bash
# List available models
python -m python list-models

# Export model weights to JSON
python -m python dump --model minimind

# Verify PyTorch ↔ C++ consistency
python -m python validate --model minimind

# Run debugging tools
python -m python debug --model minimind
python -m python debug --model minimind --layer 0
```

### 3. Using Convenience Scripts

```bash
# Build C++ inference engine
bash scripts/build_cpp.sh

# Run complete validation pipeline
bash scripts/run_validation.sh minimind

# Clean build artifacts
bash scripts/clean.sh
```

### 4. One-Click Verification ⭐

```bash
# 一键验证 PyTorch ↔ C++ 一致性
python -m python.tools.verify_consistency
```

验证工具会自动执行：
1. 编译 C++ 推理引擎
2. 导出模型权重到 `dump_minimind/`
3. 运行 PyTorch 和 C++ 推理
4. 比较输出并报告误差

**当前验证状态**: ✅ 通过 (最大误差 0.0005 < 阈值 0.001)

## 📁 Project Structure

```
CelerInfer/
├── python/                    # Core Python module
│   ├── __init__.py           # Package entry point
│   ├── __main__.py           # CLI entry point
│   ├── core/                 # Model definitions
│   ├── export/               # Weight dumping (PyTorch → JSON)
│   ├── inference/            # Inference & verification
│   ├── debug/                # Debugging & layer extraction
│   ├── validate/             # Validation & comparison
│   ├── utils/                # Common utilities
│   └── tools/                # Additional tools
│
├── cpp/                       # C++ inference engine
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── models/           # Model implementations
│   │   ├── ops/              # Tensor operations
│   │   ├── utils/
│   │   └── inference/
│   ├── include/              # Public headers
│   ├── third_party/          # Dependencies (nlohmann JSON)
│   └── build/                # Build output
│
├── models/                    # Model configs & weights
│   ├── minimind/             # MiniMind model
│   │   └── config.json       # Configuration
│   └── llama/                # LLAMA (placeholder)
│
├── dump_minimind/             # Generated outputs (gitignored)
│   ├── minimind.json         # Exported weights + config
│   ├── logits_torch.npy      # PyTorch inference output
│   └── logits_cpp.npy        # C++ inference output
│
├── scripts/                   # Convenience shell scripts
│   ├── build_cpp.sh
│   ├── run_validation.sh
│   └── clean.sh
│
├── docs/                      # Project documentation
│   ├── ARCHITECTURE.md       # Architecture & extension guide
│   ├── MODELS.md             # Supported models
│   ├── archives/             # Historical documents
│   └── legacy/               # Old debug scripts
│
├── data/                      # Test data
│   ├── input/
│   └── output/
│
└── README.md (this file)
```

## 📖 Documentation

- **[docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)** - Main documentation hub
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Detailed architecture & extension guide
- **[docs/DIRECTORY_GUIDE.md](docs/DIRECTORY_GUIDE.md)** - Quick reference & troubleshooting
- **[docs/CONSISTENCY_REPORT.md](docs/CONSISTENCY_REPORT.md)** - Full analysis & testing results
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Low-level design patterns
- **[docs/MODELS.md](docs/MODELS.md)** - Supported models list

## 🎯 Supported Models

### MiniMind ✅
- **Type**: Transformer with RoPE, RMSNorm, SiLU
- **Config**: Hidden=64, Layers=2, Heads=8, FFN=256
- **Features**: Attention, FFN, optional MoE support
- **Status**: ✅ Fully implemented and verified
- **Consistency**: PyTorch ↔ C++ max error: **0.0005** (threshold: 0.001)

### LLAMA 📋
- **Status**: Planned
- **Expected**: Support for LLAMA 2/3 models

### Qwen 📋
- **Status**: Planned

## 🔧 Core Workflows

### Workflow 1: Train → Dump → Verify → Infer

```
PyTorch Model
    ↓
python/core/minimind_model.py (Define architecture)
    ↓
python/export/minimind_dumper.py (Export to JSON)
    ↓
python/inference/minimind_forward.py (Verify against PyTorch)
    ↓
cpp/src/models/minimind.cpp (C++ Inference)
    ↓
python/validate/compare_*.py (Compare outputs)
```

### Workflow 2: One-Click Verification ⭐

```bash
# 一键验证 (推荐)
python -m python.tools.verify_consistency

# 或使用脚本
bash scripts/run_validation.sh minimind
```

**输出目录**: `dump_minimind/`
- `minimind.json` - 导出的权重和配置
- `logits_torch.npy` - PyTorch 推理结果
- `logits_cpp.npy` - C++ 推理结果

## 🛠️ Adding a New Model

See [docs/ARCHITECTURE.md#adding-a-new-model](docs/ARCHITECTURE.md#adding-a-new-model) for complete guide.

Quick summary:
1. Create `models/mymodel/config.json`
2. Implement `python/core/mymodel_model.py`
3. Implement `python/export/mymodel_dumper.py`
4. Implement `python/inference/mymodel_forward.py`
5. Register in `python/core/__init__.py`
6. Implement C++ version in `cpp/src/models/mymodel.cpp`

## 📊 Project Statistics

| Aspect | Count |
|--------|-------|
| Python Modules | 7 (core, export, inference, debug, validate, utils, tools) |
| Debug Scripts | 5+ |
| Comparison Scripts | 10+ |
| C++ Files | Modular (src/, ops/, utils/) |
| Documentation Files | 4+ |

## 🏗️ Architecture Highlights

### Python Module Organization
- **core**: Model definitions with factory pattern
- **export**: Unified weight dumping interface
- **inference**: Verification against PyTorch
- **debug**: Layer extraction and analysis
- **validate**: Comprehensive comparison tools

### C++ Design
- Header-only tensor operations for efficiency
- Factory pattern for model instantiation
- Struct-based inference (no class overhead)
- JSON+Base64 weight format for transparency

### CLI Interface
- Single entry point: `python -m python <command>`
- Subcommands: dump, validate, debug, list-models
- Model-agnostic design

## ⚙️ Build & Test

### Building C++ Engine
```bash
bash scripts/build_cpp.sh
# Or manually:
cd cpp && mkdir -p build && cd build && cmake .. && make
```

### Running Validation
```bash
# Complete validation pipeline
bash scripts/run_validation.sh minimind

# Or step by step:
python -m python dump --model minimind
python -m python validate --model minimind
```

### Cleaning Artifacts
```bash
bash scripts/clean.sh
```

## 📝 Environment

- **Python**: 3.12.11
- **C++ Standard**: C++14
- **Dependencies**: 
  - Python: torch, transformers, numpy
  - C++: nlohmann/json (included)
- **Build System**: CMake 3.16+

## 🤝 Contributing

To add a new model or feature:
1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
2. Follow the module structure
3. Add documentation
4. Test thoroughly
5. Commit with clear messages

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

## 🔗 References

- **Project Refactoring**: See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- **Architecture Details**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Model List**: See [docs/MODELS.md](docs/MODELS.md)
- **Legacy Docs**: See [docs/archives/](docs/archives/)

## 📞 Quick Links

| Resource | Location |
|----------|----------|
| CLI Entry | [python/__main__.py](python/__main__.py) |
| Model Registry | [python/core/__init__.py](python/core/__init__.py) |
| MiniMind Config | [models/minimind/config.json](models/minimind/config.json) |
| Architecture Guide | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Validation Scripts | [scripts/](scripts/) |

---

**Last Updated**: January 27, 2026  
**Version**: 0.1.0  
**Status**: ✅ Multi-model architecture ready

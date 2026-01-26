# CelerInfer Project Architecture

## Overview

CelerInfer is a modular, multi-model LLM inference framework that bridges PyTorch model development with optimized C++ inference. The project is organized to support multiple model architectures (MiniMind, LLAMA, etc.) with clear separation between core functionality, debugging, and validation.

## Directory Structure

```
CelerInfer/
├── python/                    # Python module (main implementation)
│   ├── __init__.py           # Package initialization
│   ├── __main__.py           # CLI entry point
│   ├── core/                 # Model definitions (base class + implementations)
│   ├── export/               # Weight dumping (PyTorch → JSON)
│   ├── inference/            # Inference & verification (JSON → numpy)
│   ├── debug/                # Debugging & layer extraction
│   ├── validate/             # Validation & comparison utilities
│   ├── utils/                # Common utilities (IO, config, etc.)
│   └── tools/                # Additional tools (model generation, etc.)
│
├── cpp/                       # C++ inference engine
│   ├── CMakeLists.txt        # Build configuration
│   ├── src/                  # Source code
│   │   ├── models/           # Model implementations
│   │   ├── ops/              # Tensor operations
│   │   ├── utils/            # Utilities (config, tensor, JSON)
│   │   └── inference/        # Inference engine
│   ├── include/              # Public headers
│   └── third_party/          # Dependencies (nlohmann JSON)
│
├── models/                    # Model configurations & weights
│   ├── minimind/             # MiniMind model
│   │   ├── config.json       # Configuration
│   │   ├── minimind.json     # Weights (exported)
│   │   └── README.md
│   └── llama/                # LLAMA model (placeholder)
│
├── scripts/                   # Convenience shell scripts
│   ├── build_cpp.sh          # Build C++ engine
│   ├── run_validation.sh     # Run complete validation
│   └── clean.sh              # Clean artifacts
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md       # This file
│   ├── DEBUGGING.md          # Debugging guide
│   ├── MODELS.md             # Supported models
│   └── API.md                # API reference
│
└── data/                      # Test data
    ├── input/                # Input test files
    └── output/               # Output results
```

## Design Principles

### 1. **Multi-Model Support**
- Each model has its own directory under `models/`
- Model-specific code is in dedicated modules (e.g., `minimind_model.py`, `minimind_forward.py`)
- Common interfaces defined in `core/`, `export/`, `inference/`
- Easy to add new models by following the same pattern

### 2. **Modular Python**
- **core**: Model architecture definitions
- **export**: Convert PyTorch models to JSON weights
- **inference**: Load JSON and run forward pass with verification
- **debug**: Extract intermediate outputs for analysis
- **validate**: Compare PyTorch and C++ outputs

### 3. **Unified CLI**
```bash
python -m python dump --model minimind          # Export weights
python -m python validate --model minimind      # Verify consistency
python -m python debug --model minimind         # Run debugging
python -m python list-models                    # List available models
```

### 4. **C++ Architecture**
- Base classes define interfaces
- Model-specific implementations inherit from base
- Tensor operations organized by category (attention, FFN, etc.)
- Factory pattern for model instantiation

## Workflow

### Training & Export
```
PyTorch Model → llm_minimind_model.py → minimind_dumper.py → models/minimind/minimind.json
```

### Verification & Inference
```
dump_minimind/minimind.json → minimind_forward.py → logits_torch.npy
dump_minimind/minimind.json → C++ engine        → logits_cpp.npy
                          ↓
              verify_consistency.py → 比较输出 → 报告误差
```

### One-Click Verification (推荐)
```bash
python -m python.tools.verify_consistency
```

自动执行: 编译C++ → 导出权重 → PyTorch推理 → C++推理 → 比较输出

**输出目录**: `dump_minimind/`
| 文件 | 说明 |
|------|------|
| minimind.json | 模型权重 + 配置 (Base64编码) |
| logits_torch.npy | PyTorch推理输出 |
| logits_cpp.npy | C++推理输出 |

## Adding a New Model

### Step 1: Create Model Directory
```bash
mkdir -p models/mymodel
```

### Step 2: Add Config
```json
// models/mymodel/config.json
{
  "model_type": "mymodel",
  "cpp_executable": "mymodel",
  "weights_file": "mymodel.json",
  "config": { ... }
}
```

### Step 3: Implement Python Model
```python
# python/core/mymodel_model.py
class MyModelForCausalLM(PreTrainedModel):
    ...

# python/export/mymodel_dumper.py
class MyModelDumper:
    ...

# python/inference/mymodel_forward.py
class MyModelVerifier:
    ...
```

### Step 4: Register Model
```python
# python/core/__init__.py
_MODEL_REGISTRY["mymodel"] = {
    "model_class": "mymodel_model.MyModelForCausalLM",
    "config": "models/mymodel/config.json",
}
```

### Step 5: Implement C++ Version
```cpp
// cpp/src/models/mymodel.hpp
class MyModel : public BaseModel {
    ...
};
```

## Key Features

- ✅ Multi-model support with pluggable architecture
- ✅ PyTorch ↔ C++ consistency verification
- ✅ Layer-by-layer debugging and extraction
- ✅ Comprehensive validation suite
- ✅ Unified command-line interface
- ✅ Modular C++ with base classes and factories
- ✅ Easy to extend with new models and operations

## File Relationships

| Operation | Python Module | C++ Module |
|-----------|---------------|------------|
| Define architecture | `core/minimind_model.py` | `cpp/src/models/minimind.hpp` |
| Export weights | `export/minimind_dumper.py` | - |
| Verify inference | `inference/minimind_forward.py` | `cpp/src/inference/` |
| Debug layers | `debug/minimind_debug.py` | - |
| Compare outputs | `validate/compare_*.py` | - |

## Performance Considerations

- C++ uses raw pointers and stack allocation for efficiency
- Tensor operations organized by memory access patterns
- Inline headers (`tensor_ops.hpp`) for compiler optimization
- No dynamic allocation in hot paths

## Future Enhancements

- [ ] Support for LLAMA, Qwen, and other models
- [ ] Quantization support (INT8, FP16)
- [ ] Multi-GPU inference
- [ ] Performance profiling and optimization
- [ ] Batch inference optimization

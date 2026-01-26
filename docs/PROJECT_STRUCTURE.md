# CelerInfer Project Structure Guide

This document provides a comprehensive overview of the CelerInfer project organization, module dependencies, and key workflows.

## Table of Contents

1. [Directory Layout](#directory-layout)
2. [Module Organization](#module-organization)
3. [Key Workflows](#key-workflows)
4. [Dependencies & Import Chains](#dependencies--import-chains)
5. [Adding New Models](#adding-new-models)
6. [Troubleshooting](#troubleshooting)

---

## Directory Layout

```
CelerInfer/
├── python/                          # Main Python implementation
│   ├── __init__.py                  # Package init
│   ├── __main__.py                  # Unified CLI entry point
│   ├── core/                        # Model definitions & registry
│   │   ├── __init__.py              # Model factory (get_model, list_models)
│   │   └── minimind_model.py        # MiniMind PyTorch implementation
│   ├── export/                      # Weight dumping (PyTorch → JSON)
│   │   ├── __init__.py              # Dumper factory (get_dumper, dump_model)
│   │   └── minimind_dumper.py       # MiniMind weight exporter
│   ├── inference/                   # Verification (JSON → C++/PyTorch comparison)
│   │   ├── __init__.py              # Verifier factory (get_verifier, verify_consistency)
│   │   └── minimind_forward.py      # PyTorch forward pass & MinimindVerifier class
│   ├── debug/                       # Layer extraction & debugging tools
│   │   ├── __init__.py              # Debugger factory (get_debugger)
│   │   ├── minimind_debug.py        # MiniMind layer extraction
│   │   └── debug_*.py               # Specific debug scripts
│   ├── validate/                    # Comparison & validation utilities
│   │   ├── __init__.py
│   │   └── compare_*.py             # Layer-wise comparison scripts
│   ├── utils/                       # Common utilities (IO, config parsing, etc.)
│   │   ├── __init__.py
│   │   └── [utility modules]
│   └── tools/                       # Additional tools
│       ├── __init__.py
│       └── generate_random_model.py # Random model generation for testing
│
├── cpp/                             # C++ inference engine
│   ├── CMakeLists.txt               # Build configuration
│   ├── base_line_micro.cpp          # Main C++ inference (struct-based)
│   ├── tensor_op.hpp                # Header-only tensor operations
│   ├── src/                         # Source code directory (extensible)
│   ├── include/                     # Public headers (extensible)
│   ├── third_party/                 # Dependencies
│   │   └── nlohmann/                # JSON library (json.hpp, json_fwd.hpp)
│   └── build/                       # Build artifacts (generated)
│
├── models/                          # Model configs & weights
│   ├── minimind/                    # MiniMind model
│   │   ├── config.json              # Configuration (schema)
│   │   ├── minimind.json            # Exported weights (Base64 encoded)
│   │   └── README.md
│   └── llama/                       # LLAMA model (placeholder)
│
├── scripts/                         # Shell script helpers
│   ├── build_cpp.sh                 # Compile C++ engine
│   ├── run_validation.sh            # Full validation pipeline
│   ├── clean.sh                     # Clean artifacts
│   └── benchmark.sh                 # Benchmark runner
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # Detailed architecture & extension guide
│   ├── MODELS.md                    # Supported models list
│   ├── REFACTORING_SUMMARY.md       # Recent refactoring notes
│   ├── VALIDATION_REPORT.md         # Validation results
│   ├── archives/                    # Historical documents
│   └── legacy/                      # Old debug scripts (deprecated)
│
├── data/                            # Test data
│   ├── input/                       # Input samples
│   └── output/                      # Output samples
│
├── .github/                         # GitHub configuration
│   ├── workflows/                   # CI/CD workflows
│   │   └── consistency_validation.yml  # Automated validation
│   └── copilot-instructions.md      # AI agent context
│
├── dump_minimind/                   # Runtime output directory
│   ├── minimind.json                # Dumped weights
│   ├── logits_torch.npy             # PyTorch forward output
│   └── h0_torch.npy                 # Embedding output
│
├── build/                           # Root-level build artifacts
├── .gitignore
├── LICENSE
└── README.md
```

---

## Module Organization

### `python/core/` - Model Registry & Factory

**Purpose**: Central registry for all supported models, instantiation factory.

**Key Files**:
- `__init__.py`: Defines `_MODEL_REGISTRY`, exports `get_model()`, `list_models()`, `register_model()`
- `minimind_model.py`: PyTorch model implementation (`MiniMindConfig`, `MiniMindForCausalLM`)

**Usage**:
```python
from python.core import get_model, list_models
models = list_models()  # ['minimind']
model = get_model('minimind', config_path='models/minimind/config.json')
```

---

### `python/export/` - Weight Dumping

**Purpose**: Export PyTorch model weights to JSON format with Base64 encoding.

**Key Files**:
- `__init__.py`: Factory function `get_dumper()`, `dump_model()`
- `minimind_dumper.py`: `MinimindDumper` class implementing weight export

**Export Format**: JSON with:
- `meta`: Metadata (seed, batch size, sequence length, etc.)
- `config`: Model configuration (hyperparameters)
- `inputs`: Input tensors (Base64 encoded)
- `weights`: All model weights per layer (Base64 encoded)
- `rope`: Precomputed RoPE cos/sin values

**Usage**:
```python
from python.export import get_dumper
dumper = get_dumper('minimind')
dumper.dump(model, output_dir='models/minimind')
```

---

### `python/inference/` - Verification & Validation

**Purpose**: Load JSON weights and run PyTorch forward pass for consistency verification.

**Key Files**:
- `__init__.py`: Factory function `get_verifier()`, `verify_consistency()`
- `minimind_forward.py`: `MinimindVerifier` class wrapping the main verification logic

**Verification Steps**:
1. Load JSON file containing model config, weights, inputs
2. Instantiate PyTorch model from config
3. Load weights from JSON into model
4. Run forward pass with timing
5. Save outputs (logits, embeddings) for C++ comparison

**Usage**:
```python
from python.inference import verify_consistency
verify_consistency('minimind')  # Runs full validation, saves outputs
```

---

### `python/debug/` - Layer-wise Debugging

**Purpose**: Extract and inspect intermediate layer outputs for debugging.

**Key Files**:
- `__init__.py`: Factory `get_debugger()`
- `minimind_debug.py`: `MiniMindDebugger` class
- `debug_*.py`: Specialized scripts for specific layer analysis

**Features**:
- Extract embedding outputs
- Compare layer-wise attention/FFN outputs
- Analyze residual connections
- QKV projection debugging

**Usage**:
```python
from python.debug import get_debugger
debugger = get_debugger('minimind')
debugger.extract_layer(0)  # Extract layer 0
debugger.debug_all()       # Extract all layers
```

---

### `python/validate/` - Comparison Tools

**Purpose**: Compare PyTorch and C++ outputs layer-by-layer.

**Key Files**:
- `compare_*.py`: Comparison scripts for logits, attention, FFN, etc.
- `compute_full_attention.py`: Reference attention implementation

**Functions**:
- Load outputs from both backends
- Compute statistical differences (max, mean, correlation)
- Generate reports

---

### `cpp/` - C++ Inference Engine

**Purpose**: Optimized C++ implementation for fast inference.

**Key Files**:
- `base_line_micro.cpp`: Main inference loop (struct-based, no classes)
- `tensor_op.hpp`: Header-only tensor operations library
- `CMakeLists.txt`: Build configuration

**Design Patterns**:
- Struct-based (not OOP): Raw `const float*` pointers for weights
- Header-only ops: Generic tensor operations with negative indexing
- Manual memory management: Stack-based or pre-allocated buffers
- JSON loading: Uses `nlohmann::json` to parse weight manifests

**Build**:
```bash
mkdir -p build && cd build
cmake ..
make
./base_line_micro
```

---

## Key Workflows

### Workflow 1: Train & Export

```
PyTorch Model (training/finetune)
    ↓
python/core/minimind_model.py (Define architecture)
    ↓
Instantiate MiniMindForCausalLM
    ↓
python/export/minimind_dumper.py (Dump weights to JSON)
    ↓
models/minimind/minimind.json (Exported weights)
```

**CLI Command**:
```bash
python -m python dump --model minimind --output models/minimind
```

---

### Workflow 2: Verify Consistency

```
models/minimind/minimind.json (Weights)
    ↓
python/inference/minimind_forward.py (Load & verify)
    ↓
PyTorch forward pass
    ↓
dump_minimind/logits_torch.npy (Reference output)
    ↓
cpp/base_line_micro.cpp (C++ forward pass)
    ↓
dump_minimind/logits_cpp.bin (C++ output)
    ↓
python/validate/compare_*.py (Compare & report)
```

**CLI Command**:
```bash
python -m python validate --model minimind
```

---

### Workflow 3: Debug Layer Outputs

```
Model instance
    ↓
python/debug/minimind_debug.py
    ↓
Extract layer N outputs
    ↓
dump_minimind/layer_N_*.npy
    ↓
Analyze/compare with C++ outputs
```

**CLI Command**:
```bash
python -m python debug --model minimind --layer 0
```

---

### Workflow 4: Run Full CI/CD

```
Git push
    ↓
.github/workflows/consistency_validation.yml
    ↓
Build C++
    ↓
Run python -m python validate --model minimind
    ↓
Report results
```

---

## Dependencies & Import Chains

### Python Import Hierarchy

```
python/__main__.py (CLI entry)
    ├─→ python.core (get_model, list_models)
    ├─→ python.export (dump_model)
    ├─→ python.inference (verify_consistency)
    ├─→ python.debug (get_debugger)
    └─→ python.validate (compare functions)
        ↓
        └─→ python.core.minimind_model (PyTorch layers)
```

### C++ Dependencies

```
base_line_micro.cpp
    ├─→ tensor_op.hpp (tensor operations)
    ├─→ third_party/nlohmann/json.hpp (JSON parsing)
    └─→ models/minimind/minimind.json (weights manifest)
```

---

## Adding New Models

To add a new model (e.g., `llama`):

### Step 1: Create Model Config

**File**: `models/llama/config.json`
```json
{
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  ...
}
```

### Step 2: Implement Python Model

**File**: `python/core/llama_model.py`
```python
class LlamaConfig:
    # Config class
    pass

class LlamaForCausalLM(torch.nn.Module):
    # Model implementation
    pass
```

### Step 3: Implement Dumper

**File**: `python/export/llama_dumper.py`
```python
class LlamaDumper:
    def dump(self, model, output_dir):
        # Export weights to JSON
        pass
```

### Step 4: Implement Verifier

**File**: `python/inference/llama_forward.py`
```python
class LlamaVerifier:
    def verify(self, config_path):
        # Load JSON, run forward pass, save outputs
        pass
```

### Step 5: Register Model

**File**: `python/core/__init__.py`
```python
_MODEL_REGISTRY["llama"] = {
    "model_class": "llama_model.LlamaForCausalLM",
    "config": "models/llama/config.json",
}
```

### Step 6: Update Factories

Update `__init__.py` in `export/`, `inference/`, `debug/` to handle 'llama'.

### Step 7: Implement C++ Version

**File**: `cpp/src/models/llama.cpp`
```cpp
struct llama_config { ... };
void llama_forward(...) { ... }
```

---

## Troubleshooting

### Issue: Import Error - `MinimindVerifier not found`

**Cause**: The verifier class is not exposed in the module.

**Fix**: Check `python/inference/minimind_forward.py` exports `MinimindVerifier` class.

```python
class MinimindVerifier:
    def verify(self, config_path):
        main()
        return True
```

### Issue: JSON Not Found

**Cause**: `verify_consistency()` looks for weights at default path.

**Fix**: Pass explicit path or set `JSON_PATH` environment variable:
```bash
export JSON_PATH=models/minimind/minimind.json
python -m python validate --model minimind
```

### Issue: C++ Build Fails

**Cause**: CMakeLists.txt configuration issue.

**Fix**: 
```bash
cd cpp/build
cmake -DCMAKE_CXX_FLAGS="-O2 -g" ..
make
```

### Issue: Output Mismatch Between C++ and PyTorch

**Cause**: Numerical precision, RoPE scaling, or weight loading differences.

**Solution**: 
1. Run `python -m python debug --model minimind --layer 0` to extract intermediate outputs
2. Compare layer-by-layer with `python/validate/compare_*.py` scripts
3. Check weight loading in both implementations matches exactly

---

## Statistics

| Aspect | Value |
|--------|-------|
| Python Modules | 7 (core, export, inference, debug, validate, utils, tools) |
| Python Files | ~20 |
| C++ Source Files | 1 (base_line_micro.cpp) + extensible |
| Header-only Libraries | 1 (tensor_op.hpp) |
| Third-party Dependencies | nlohmann/json |
| Supported Models | minimind (full), llama (planned) |
| Tests | GitHub Actions CI/CD workflow |

---

## Next Steps

1. **Model Expansion**: Add LLAMA, Qwen, or other architectures
2. **Performance Optimization**: Implement kernel fusions, batching optimizations
3. **Multi-Backend Support**: Add CUDA, Metal, or other acceleration
4. **Serving Integration**: REST API wrapper for inference
5. **Quantization**: Int8/Int4 quantization support


# CelerInfer Directory & Consistency Guide

This document consolidates all essential information about the CelerInfer project structure, consistency verification, and quick reference.

## Quick Reference

| What? | Command | Location |
|-------|---------|----------|
| **List models** | `python -m python list-models` | [python/core/__init__.py](python/core/__init__.py) |
| **Dump weights** | `python -m python dump --model minimind --output models/minimind` | [python/export/minimind_dumper.py](python/export/minimind_dumper.py) |
| **Validate (PyTorch ↔ C++)** | `python -m python validate --model minimind` | [python/inference/minimind_forward.py](python/inference/minimind_forward.py) |
| **Debug (extract layers)** | `python -m python debug --model minimind --layer 0` | [python/debug/minimind_debug.py](python/debug/minimind_debug.py) |
| **Build C++** | `bash scripts/build_cpp.sh` or `cd cpp && cmake . && make` | [cpp/CMakeLists.txt](cpp/CMakeLists.txt) |
| **Compare outputs** | `cd python/validate && python compare_logits.py` | [python/validate/](python/validate/) |

---

## Consistency Verification Architecture

### Data Flow

```
1. Models & Configs
   ↓
models/minimind/config.json
models/minimind/minimind.json (exported weights)
   ↓
2. Python Forward Pass
   ↓
python/core/minimind_model.py        ← PyTorch definition
python/export/minimind_dumper.py     ← Export to JSON
python/inference/minimind_forward.py ← Load JSON + forward
   ↓
3. Outputs Saved
   ↓
dump_minimind/logits_torch.npy       ← PyTorch logits
dump_minimind/h0_torch.npy           ← Embeddings
   ↓
4. C++ Forward Pass
   ↓
cpp/base_line_micro.cpp              ← Load JSON + infer
cpp/tensor_op.hpp                    ← Tensor ops
   ↓
5. Outputs Compared
   ↓
python/validate/compare_logits.py    ← Max diff, correlation, etc.
   ↓
✓ Reports consistency status
```

---

## Module Reference

### `python/core/` - Model Registry

**Files**:
- `__init__.py` - Exports `get_model(name)`, `list_models()`, `register_model()`
- `minimind_model.py` - PyTorch implementation

**Usage**:
```python
from python.core import get_model, list_models
model = get_model('minimind')
```

**Config Format** (`models/minimind/config.json`):
```json
{
  "config": {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    ...
  }
}
```

---

### `python/export/` - Weight Dumping

**Files**:
- `__init__.py` - Exports `dump_model(model_name, model, output_dir)`
- `minimind_dumper.py` - `MinimindDumper` class

**Output Format** (`models/minimind/minimind.json`):
```json
{
  "meta": {"seed": 123, "B": 2, "S": 5},
  "config": { ... model hyperparameters ... },
  "inputs": {"input_ids": {...}, "attention_mask": {...}},
  "weights": {
    "tok_embedding": { "shape": [...], "dtype": "float32", "encoding": "base64", "data": "..." },
    "layers": [...]
  },
  "rope": {"cos": {...}, "sin": {...}}
}
```

**Usage**:
```bash
python -m python dump --model minimind --output models/minimind
```

---

### `python/inference/` - Verification

**Files**:
- `__init__.py` - Exports `verify_consistency(model_name)`
- `minimind_forward.py` - `MinimindVerifier` class with PyTorch forward pass

**Verification Steps**:
1. Load JSON manifest (config + weights + inputs)
2. Instantiate PyTorch model from config
3. Load all weights into model from Base64 data
4. Run forward pass with timing
5. Save logits & embeddings

**Outputs**:
- `dump_minimind/logits_torch.npy` - Reference logits (B, S, V)
- `dump_minimind/h0_torch.npy` - Embedding outputs (B, S, H)

**Usage**:
```bash
python -m python validate --model minimind
```

**Customization**:
```bash
export JSON_PATH="path/to/custom.json"
export WARMUP=5              # Number of warmup iterations
export JSON_PREVIEW_N=32     # Preview values in JSON
python -m python validate --model minimind
```

---

### `python/debug/` - Layer Extraction

**Files**:
- `__init__.py` - Exports `get_debugger(model_name)`
- `minimind_debug.py` - `MiniMindDebugger` class
- `debug_*.py` - Specialized layer analysis scripts

**Features**:
- Extract embedding outputs
- Extract layer-wise outputs
- Compare intermediate values
- Analyze attention patterns

**Usage**:
```bash
python -m python debug --model minimind --layer 0
python -m python debug --model minimind              # All layers
```

---

### `python/validate/` - Comparison Tools

**Files**:
- `compare_logits.py` - Compare final logits (max diff, correlation)
- `compare_attention_*.py` - Attention mechanism comparison
- `compare_ffn.py` - FFN layer comparison
- `compute_full_attention.py` - Reference attention impl

**Usage**:
```bash
cd python/validate
python compare_logits.py          # Compare PyTorch vs C++ logits
python compare_attention_scores.py
```

**Output** (typical):
```
Max diff: 5.2e-4
Mean diff: 1.1e-4
Correlation: 0.9987
```

---

## C++ Implementation

### Files

| File | Purpose |
|------|---------|
| [cpp/base_line_micro.cpp](cpp/base_line_micro.cpp) | Main inference loop + weight loading |
| [cpp/tensor_op.hpp](cpp/tensor_op.hpp) | Header-only tensor operations |
| [cpp/CMakeLists.txt](cpp/CMakeLists.txt) | Build configuration |
| [cpp/third_party/nlohmann/json.hpp](cpp/third_party/nlohmann/json.hpp) | JSON parsing |

### Design Patterns

**Struct-based (not OOP)**:
```cpp
struct minimind_layer_weights {
  const float* w_q = nullptr;   // Query projection
  const float* w_k = nullptr;   // Key projection
  const float* w_v = nullptr;   // Value projection
  // ...
};
```

**Header-only tensor ops** (`tensor_op.hpp`):
```cpp
void silu(float* out, const int64_t* out_shape, 
          const float* a, const int64_t* a_shape);
void matmul_3d(float* out, const int64_t* o_shape,
               const float* a, const int64_t* a_shape,
               const float* b, const int64_t* b_shape);
```

**Negative indexing convention**:
```cpp
// 3D tensor: (batch, seq, hidden)
// Negative indices: a2 = batch, a1 = seq, a0 = hidden
int64_t offset = (i2 * a1 + i1) * a0 + i0;
```

### Build

```bash
cd cpp
mkdir build && cd build
cmake ..
make
./base_line_micro
```

---

## Workflow Examples

### Workflow 1: Full Consistency Check

```bash
# 1. Dump model weights to JSON
python -m python dump --model minimind --output models/minimind

# 2. Run PyTorch forward + save outputs
python -m python validate --model minimind

# 3. Run C++ inference (loads JSON, runs forward, saves logits_cpp.bin)
./cpp/build/base_line_micro

# 4. Compare outputs
cd python/validate
python compare_logits.py
```

### Workflow 2: Debug Specific Layer

```bash
# 1. Extract layer 0 from both backends
python -m python debug --model minimind --layer 0

# 2. Examine generated layer outputs
ls -la dump_minimind/layer_0_*.npy

# 3. Load and compare in Python
python python/validate/compare_intermediates.py
```

### Workflow 3: CI/CD Validation

Automated via [.github/workflows/consistency_validation.yml](.github/workflows/consistency_validation.yml):

1. Trigger on push
2. Build C++ engine
3. Run `python -m python validate --model minimind`
4. Compare logits
5. Report results

---

## Directory Structure Reference

```
CelerInfer/
│
├── python/                          # Main Python implementation
│   ├── __init__.py
│   ├── __main__.py                  ← CLI entry point
│   │
│   ├── core/                        # Model registry & definitions
│   │   ├── __init__.py              ← get_model(), list_models()
│   │   └── minimind_model.py        ← MiniMindForCausalLM (PyTorch)
│   │
│   ├── export/                      # Weight dumping
│   │   ├── __init__.py
│   │   └── minimind_dumper.py       ← MinimindDumper class
│   │
│   ├── inference/                   # Verification
│   │   ├── __init__.py
│   │   └── minimind_forward.py      ← MinimindVerifier class
│   │
│   ├── debug/                       # Layer extraction
│   │   ├── __init__.py
│   │   ├── minimind_debug.py
│   │   └── debug_*.py
│   │
│   ├── validate/                    # Comparison tools
│   │   ├── __init__.py
│   │   ├── compare_logits.py
│   │   ├── compare_attention_*.py
│   │   ├── compare_ffn.py
│   │   └── compute_full_attention.py
│   │
│   ├── utils/                       # Common utilities
│   │   └── __init__.py
│   │
│   └── tools/                       # Additional tools
│       ├── __init__.py
│       └── generate_random_model.py
│
├── cpp/                             # C++ inference
│   ├── base_line_micro.cpp          ← Main inference loop
│   ├── tensor_op.hpp                ← Tensor operations (header-only)
│   ├── CMakeLists.txt               ← Build config
│   ├── include/                     # Public headers
│   ├── src/                         # Source code (extensible)
│   ├── third_party/
│   │   └── nlohmann/                # JSON library
│   └── build/                       # Build artifacts (generated)
│
├── models/                          # Model configs & weights
│   ├── minimind/
│   │   ├── config.json              ← Model hyperparameters
│   │   ├── minimind.json            ← Exported weights (Base64)
│   │   └── README.md
│   └── llama/                       # Placeholder
│
├── scripts/                         # Shell script helpers
│   ├── build_cpp.sh
│   ├── run_validation.sh
│   ├── clean.sh
│   └── benchmark.sh
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              ← Detailed architecture
│   ├── MODELS.md                    ← Supported models
│   ├── REFACTORING_SUMMARY.md
│   ├── VALIDATION_REPORT.md
│   ├── archives/
│   └── legacy/
│
├── dump_minimind/                   # Runtime outputs
│   ├── minimind.json                ← Exported weights
│   ├── logits_torch.npy             ← PyTorch outputs
│   ├── h0_torch.npy
│   └── logits_cpp.bin               ← C++ outputs
│
├── data/                            # Test data
│   ├── input/
│   └── output/
│
├── .github/
│   ├── workflows/
│   │   └── consistency_validation.yml ← CI/CD
│   └── copilot-instructions.md
│
├── PROJECT_STRUCTURE.md             ← Comprehensive guide
├── DIRECTORY_GUIDE.md               ← This file
├── README.md
└── LICENSE
```

---

## Troubleshooting

### Issue: "Cannot import MinimindVerifier"

**Fix**: Verify `python/inference/minimind_forward.py` has the class:
```python
class MinimindVerifier:
    def verify(self, config_path):
        main()
        return True
```

### Issue: "Config file not found"

**Fix**: Check config exists at `models/minimind/config.json` or pass explicit path:
```python
from python.core import get_model
model = get_model('minimind', config_path='custom/path/config.json')
```

### Issue: JSON parsing error in C++

**Fix**: Validate JSON format:
```bash
python -c "import json; json.load(open('models/minimind/minimind.json'))"
```

### Issue: C++ logits don't match PyTorch

**Solution**:
1. Check weight loading is identical (same Base64 decoding)
2. Compare embedding outputs: `python -m python debug --model minimind --layer 0`
3. Compare layer-by-layer: `cd python/validate && python compare_intermediates.py`
4. Check RoPE cos/sin values are loaded correctly

---

## Key Statistics

- **Python modules**: 7 (core, export, inference, debug, validate, utils, tools)
- **C++ files**: 1 main + extensible (base_line_micro.cpp)
- **Header-only libs**: tensor_op.hpp
- **Supported models**: minimind (full), llama (planned)
- **Test coverage**: GitHub Actions CI/CD workflow
- **Code organization**: Factory pattern, modular design

---

## Next Steps for Extension

1. **Add LLAMA model**:
   - Create `python/core/llama_model.py` (PyTorch)
   - Create `python/export/llama_dumper.py`
   - Create `python/inference/llama_forward.py`
   - Implement C++ version in `cpp/src/models/llama.cpp`

2. **Add quantization**:
   - Int8/Int4 weight quantization
   - Update dumper to quantize weights
   - Update C++ inference for quantized ops

3. **Add CUDA backend**:
   - CUDA kernel implementations in `cpp/src/ops/cuda/`
   - Runtime backend selection

4. **Add serving layer**:
   - FastAPI REST API wrapper
   - Batch inference support
   - Model caching

---

## Files Generated at Runtime

These files are created during validation/debug runs:

| File | Content | Size | Purpose |
|------|---------|------|---------|
| `dump_minimind/logits_torch.npy` | PyTorch logits | ~40KB | Reference output |
| `dump_minimind/h0_torch.npy` | Embeddings | ~10KB | Debug intermediate |
| `dump_minimind/logits_cpp.bin` | C++ logits (binary) | ~40KB | C++ output |
| `models/minimind/minimind.json` | Exported weights | ~500KB | Weight manifest |

---

## Summary

CelerInfer provides a **unified, modular framework** for:
- ✓ PyTorch model definition & training
- ✓ Weight export to human-readable JSON (Base64)
- ✓ Consistency verification (PyTorch ↔ C++)
- ✓ Layer-wise debugging & comparison
- ✓ Extensible architecture for multiple models
- ✓ Automated CI/CD validation

The **consistency workflow** ensures bit-exact or near-exact matching between PyTorch and C++ implementations, critical for production inference deployment.


# CelerInfer - AI Agent Instructions

## Project Overview
CelerInfer is a **modular, multi-model hybrid C++/Python LLM inference framework**. Currently implements MiniMind baseline micro with support for extensible architecture. The project bridges PyTorch model training/verification with optimized C++ tensor operations for efficient inference.

## Project Structure (Post-Refactoring)

### Core Directories
- **[python/](python/)** - Main Python implementation
  - `core/` - Model definitions and interfaces
  - `export/` - Weight dumping (PyTorch → JSON)
  - `inference/` - Inference and verification
  - `debug/` - Debugging and layer extraction
  - `validate/` - Comparison and validation tools
  - `__main__.py` - Unified CLI entry point

- **[cpp/](cpp/)** - C++ inference engine
  - `src/models/` - Model implementations
  - `src/ops/` - Tensor operations
  - `include/` - Public headers
  - `tensor_op.hpp` - Header-only tensor lib

- **[models/](models/)** - Model configs and weights
  - `minimind/` - MiniMind config.json + weights
  - `llama/` - Placeholder for future models

- **[scripts/](scripts/)** - Convenience shell scripts
  - `build_cpp.sh` - Compile C++
  - `run_validation.sh` - Full validation pipeline
  - `clean.sh` - Clean artifacts

- **[docs/](docs/)** - Documentation
  - `ARCHITECTURE.md` - Detailed architecture guide
  - `MODELS.md` - Supported models list

## Unified CLI Interface

```bash
# List models
python -m python list-models

# Dump weights
python -m python dump --model minimind

# Validate consistency
python -m python validate --model minimind

# Debug (layer extraction, etc.)
python -m python debug --model minimind [--layer N]
```

## Core Workflow: Train → Dump → Verify → Infer
1. **Python (PyTorch)**: `python/core/minimind_model.py` - Model definition
2. **Dump to JSON**: `python/export/minimind_dumper.py` - Weights → Base64 JSON
3. **Forward Verify**: `python/inference/minimind_forward.py` - Verify against PyTorch
4. **C++ Inference**: `cpp/src/models/minimind.cpp` - Optimized C++ inference

### Key File Purposes (Updated Locations)
- [cpp/src/ops/tensor_ops.hpp](cpp/src/ops/tensor_ops.hpp) - Header-only tensor ops library
- [cpp/src/models/minimind.cpp](cpp/src/models/minimind.cpp) - MiniMind inference
- [python/core/minimind_model.py](python/core/minimind_model.py) - PyTorch MiniMind
- [python/export/minimind_dumper.py](python/export/minimind_dumper.py) - Weight export
- [models/minimind/config.json](models/minimind/config.json) - Model configuration

## Critical Design Conventions

### Tensor Shape Convention (C-style Negative Indexing)
All tensor operations in `tensor_op.hpp` use **negative indexing** for dimensions:
- `a0` = rightmost dimension (stride=1, contiguous)
- `a1` = second from right
- `a2`, `a3`, `a4` = further left

Example: 3D offset = `(i2 * a1 + i1) * a0 + i0`

**Why**: Simplifies generic offset calculations; matches PyTorch tensor layout.

### Struct-Based Inference (No Classes)
`base_line_micro.cpp` uses plain C structs with raw `const float*` pointers:
```cpp
struct minimind_layer_weights {
  const float* w_q = nullptr;   // (H*D, hidden)
  const float* w_k = nullptr;   // (KVH*D, hidden)
  // ...
};
```
**Why**: Explicit memory layout control; avoids C++ class overhead; matches C inference patterns.

### JSON + Base64 Format
Model weights stored in JSON with Base64 encoding:
- Enables human inspection of tensor metadata (shape, dtype, preview values)
- Data field is Base64 for binary efficiency
- Preview shows first N values (env var `JSON_PREVIEW_N`)

## Developer Workflows

### Using Unified CLI
```bash
# List available models
python -m python list-models

# Build C++ (from scripts)
bash scripts/build_cpp.sh

# Dump model weights
python -m python dump --model minimind

# Validate consistency
python -m python validate --model minimind

# Run debugging
python -m python debug --model minimind --layer 0

# One-click full validation
bash scripts/run_validation.sh minimind
```

### Environment
- Python 3.12.11 (see README.md)
- Conda environment: `conda create -n CelerInfer python=3.12.11`
- C++14 standard (CMakeLists.txt)
- Dependencies: torch, transformers, numpy, nlohmann/json (third_party/)

## Adding New Models

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#adding-a-new-model) for detailed instructions:

1. Create `models/mymodel/config.json`
2. Implement Python model in `python/core/mymodel_model.py`
3. Implement dumper in `python/export/mymodel_dumper.py`
4. Implement verifier in `python/inference/mymodel_forward.py`
5. Register in `python/core/__init__.py`
6. Implement C++ version in `cpp/src/models/mymodel.cpp`

## Model Features

### Supported Configurations
- **Base**: Transformer with RoPE, RMSNorm, SiLU activation
- **Flash Attention**: Optional flag for optimized attention
- **Mixture of Experts (MoE)**: Top-k routing with shared experts
- **Rope Scaling**: YaRN-style rope for context length extension

### Key Hyperparameters (MiniMindConfig)
- `hidden_size`, `num_hidden_layers`, `num_attention_heads`
- `num_key_value_heads` - for multi-head attention
- `intermediate_size` - FFN expansion
- `rope_theta`, `max_position_embeddings`
- MoE: `use_moe`, `num_experts_per_tok`, `n_routed_experts`, `n_shared_experts`

## Integration Points

### Python ↔ C++ Data Flow
1. PyTorch model → `llm_minimind_dump.py` creates JSON manifest with weights
2. JSON parsed in C++ via `nlohmann/json` (third_party/nlohmann/)
3. Raw float pointers passed to `tensor_op.hpp` functions
4. Results verified against PyTorch forward pass in `llm_minimind_forward.py`

### Attention Mechanism
- Group Query Attention (KV heads < Q heads)
- RoPE applied in forward pass
- Optional Flash Attention acceleration

### FFN + MoE
- Standard: Dense gate+up+down projections
- MoE: Expert routing layer with auxiliary loss

## Common Patterns for New Features

### Adding Tensor Operations
1. Add function to `tensor_op.hpp` with shape parameters using negative indexing
2. Follow naming: `<op_name>(float* out, int64_t o*, const float* a, int64_t a*, ...)`
3. Test in C++ first, then verify against PyTorch implementation

### Adding Model Weights
1. Update `minimind_layer_weights` struct in `base_line_micro.cpp`
2. Add corresponding field in `llm_minimind_model.py`
3. Update `llm_minimind_dump.py` to export new weight
4. Update weight loading in `llm_minimind_forward.py`

### Config Changes
1. Add to `MiniMindConfig` in `llm_minimind_model.py`
2. Add to `minimind_config` struct in `base_line_micro.cpp`
3. Update JSON serialization in `_cfg_to_json_dict()` (llm_minimind_dump.py)
4. Update config loading in `_build_cfg()` (llm_minimind_forward.py)

## Gotchas & Important Notes

- **Stride-1 Dimension**: Always rightmost (`a0`); offset calculations rely on this
- **Forward Pass Timing**: `llm_minimind_forward.py` excludes JSON loading time but includes warmup iterations
- **Test File**: `cpp/test.cpp` is placeholder code (divides by zero); not part of build
- **Conda Environment**: Only Python 3.12.11 specified in README; lock exact versions if stability needed

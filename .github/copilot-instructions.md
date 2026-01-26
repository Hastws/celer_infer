# CelerInfer - AI Agent Instructions

## Project Overview
CelerInfer is a hybrid C++/Python LLM inference framework that implements a "baseline micro" implementation of the MiniMind model. The project bridges PyTorch model training/verification with raw C++ tensor operations for inference optimization.

## Architecture Pattern

### Core Workflow: Train → Dump → Verify → Infer
1. **Python (PyTorch)**: Define model architecture in `llm_minimind_model.py` (inherits `PreTrainedModel`)
2. **Dump to JSON**: `llm_minimind_dump.py` extracts weights from trained model → Base64-encoded JSON
3. **Forward Verify**: `llm_minimind_forward.py` loads JSON, validates inference output against PyTorch
4. **C++ Inference**: `base_line_micro.cpp` implements the actual inference loop with raw float pointers

### Key File Purposes
- [cpp/tensor_op.hpp](cpp/tensor_op.hpp) - Header-only library: 2397 lines of tensor ops (matmul, attention, norm, etc.)
- [cpp/base_line_micro.cpp](cpp/base_line_micro.cpp) - Main inference loop using tensor ops; defines `minimind_config` and layer weight structs
- [script/llm_minimind_model.py](script/llm_minimind_model.py) - PyTorch model definition with MoE support, RoPE, attention mechanisms
- [script/llm_minimind_dump.py](script/llm_minimind_dump.py) - Export weights to JSON with Base64 encoding

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

### Build C++ Code
```bash
cd cpp
mkdir -p build && cd build
cmake .. && make
./base_line_micro
```

### Dump Model Weights
```bash
cd script
python llm_minimind_dump.py
# Output: dump_minimind/minimind.json
```

### Verify Against PyTorch
```bash
export JSON_PATH=dump_minimind/minimind.json
export DUMP_DIR=./outputs
export WARMUP=10
python llm_minimind_forward.py
```

### Environment
- Python 3.12.11 (see README.md)
- Conda environment: `conda create -n CelerInfer python=3.12.11`
- C++14 standard (CMakeLists.txt)
- Dependencies: torch, transformers, numpy, nlohmann/json (third_party/)

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

# Debugging Logits Mismatch Between PyTorch and C++

## Problem Statement
PyTorch and C++ implementations produce different logits for the same input and model weights:
- **PyTorch logits**: Mean=0.0054, Std=0.177, Range=[-0.517, 1.191]
- **C++ logits**: Mean=0.0102, Std=0.201, Range=[-0.504, 1.463]
- **Correlation**: 0.705 (moderate, but significant divergence)
- **Mean absolute difference**: 0.114
- **Max absolute difference**: 0.685

## Key Finding: Embeddings are Identical ✓

**Successfully validated:**
- Embedding weights loaded correctly from JSON (min=-0.077, max=0.076)
- Input IDs loaded correctly: [126, 109, 126, 66, 92, 98, 102, 17, 83, 106]
- Embedding lookup operation works identically in both implementations
- H0 (embedding output): PyTorch and C++ match perfectly (RMSE=0.0 after dropout)

This means **the problem is NOT in the embedding layer**. The divergence occurs in the transformer layers:
1. RMSNorm operations
2. Attention computation (q/k/v projections, matmul, masking, RoPE)
3. FFN (gate/up/down projections)

## Possible Root Causes

### 1. **RoPE (Rotary Position Embeddings) Mismatch**
- RoPE requires careful handling of coordinate dimensions
- Tensor shape convention: C++ uses negative indexing (rightmost=a0)
- PyTorch uses standard positive indexing
- **Action**: Compare RoPE outputs for layer 0

### 2. **Attention Masking**
- C++ receives `attn_mask` as uint8: all 1s for valid positions
- Need to verify masking is applied correctly (should disable attention to padding)
- **Action**: Check attention weights before softmax

### 3. **RMSNorm Computation**
- Formula: `x * (gamma / sqrt(mean(x^2) + eps))`
- Could have epsilon handling or weight broadcast issues
- **Action**: Compare RMSNorm output for first layer

### 4. **Tensor Layout/Strides**
- C++ assumes C-style negative indexing for offsets
- PyTorch may have different contiguity assumptions
- **Action**: Verify tensor shapes at each operation

### 5. **FFN Gate/Activation**
- SiLU activation: `x * sigmoid(x)` 
- Gate mechanism: `(gate * up) ⊗ x → down`
- **Action**: Compare FFN outputs

## Debugging Strategy

To identify which operation causes divergence:

### Step 1: Compare After First RMSNorm
```cpp
// In C++: After embedding and before layer 0
rms_norm_lastdim(h0, B*S, hidden, layer_w[0].rms_attn, hidden, eps, h1_normed);
// Save h1_normed to file
```

```python
# In PyTorch: After embedding
h = model.model.dropout(model.model.embed_tokens(input_ids_t))
h_normed = model.model.layers[0].input_layernorm(h)
# Compare with C++ output
```

### Step 2: Compare Attention Output
```cpp
// After attention in C++
// Save ws->h1 (result of attention + residual)
```

```python
# In PyTorch: After attention in layer 0
# Use hooks or manual layer execution
```

### Step 3: Compare FFN Output
Compare outputs after the FFN in layer 0.

## Code Changes Made

1. **Fixed Base64 Decoder** ✓
   - Was corrupting all weight data (values e+29)
   - Rewritten with cleaner bit-by-bit logic

2. **Fixed Type Conversion** ✓
   - int32 input_ids and uint8 attention_mask were being converted through float32
   - Now using direct binary memcpy

3. **Separated Embedding Verification** ✓
   - Added debug output to save embedding weights and h0 embedding output
   - Confirmed both match perfectly

4. **Infrastructure for Debugging** ✓
   - Global variable `g_h0_embedding` to save embedding output
   - Debug output showing h0 stats after embedding lookup
   - Python scripts to load and compare binary outputs

## File Locations

**Key files for debugging:**
- [cpp/base_line_micro.cpp](cpp/base_line_micro.cpp) - Main forward pass, contains `minimind_forward_infer()`
- [cpp/tensor_op.hpp](cpp/tensor_op.hpp) - Tensor operations: matmul, attention, norm, etc.
- [script/llm_minimind_forward.py](script/llm_minimind_forward.py) - PyTorch reference implementation
- [script/llm_minimind_model.py](script/llm_minimind_model.py) - Model architecture definition

**Generated comparison files:**
- `dump_minimind/h0_torch.npy` - PyTorch embedding output (verified correct)
- `dump_minimind/h0_cpp.npy` - C++ embedding output (verified correct, matches PyTorch)
- `dump_minimind/logits_torch.npy` - PyTorch final logits
- `dump_minimind/logits_cpp.npy` - C++ final logits (correlates 0.705 with PyTorch)

## Next Steps

1. **Modify C++ to save intermediate outputs** after each operation in layer 0
2. **Add Python helper to load and compare** intermediate outputs
3. **Binary search** through operations to identify which one diverges
4. **Fix the divergent operation** in either C++ or PyTorch to match the other

## Hypothesis

The correlation of 0.705 suggests the C++ implementation is computing something similar but with systematic differences. This is most likely:
- **RoPE scaling wrong** by some factor
- **Attention softmax scaling** incorrect (missing temperature or scale factor)
- **Layer normalization** off by numerical factor

The fact that values can be opposite signs (first logit: 0.149 vs -0.067) suggests it's not just a scale factor.

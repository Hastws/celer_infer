# CelerInfer Random Model Benchmarking Guide

This guide explains the new random model generation and benchmarking workflow that allows you to quickly test inference performance without needing trained weights.

## Quick Start

### 1. Generate Random Model

```bash
python script/generate_random_model.py \
    --hidden 64 \
    --layers 2 \
    --heads 8 \
    --kvh 2 \
    --vocab 128 \
    --seq-len 5 \
    --batch-size 2 \
    --seed 123 \
    --output dump_minimind/minimind.json
```

This creates a JSON file with:
- Random weight matrices (using Base64 encoding)
- Model configuration (hidden size, layers, heads, etc.)
- Input tensors (input_ids, attention_mask)
- RoPE cache (cos/sin values)

### 2. Run PyTorch Forward Pass

```bash
export JSON_PATH=dump_minimind/minimind.json
export DUMP_DIR=dump_minimind
export WARMUP=5

python script/llm_minimind_forward.py
```

**Output:**
- Prints forward pass timing (in milliseconds, excluding JSON loading)
- Saves logits to `dump_minimind/logits_torch.npy`
- Warmup iterations help stabilize timing measurements

### 3. Run C++ Inference

```bash
cd cpp
mkdir -p build && cd build
cmake ..
make

./base_line_micro ../dump_minimind/minimind.json ../dump_minimind
```

**Output:**
- Prints C++ forward pass timing
- Saves logits to `dump_minimind/logits_cpp.npy`
- Can compare with PyTorch results for verification

## One-Step Benchmarking

Run all three steps at once:

```bash
bash benchmark.sh [hidden] [layers] [heads] [kvh] [vocab] [seq_len] [batch_size] [seed]
```

**Examples:**

```bash
# Small model (default)
bash benchmark.sh

# Medium model
bash benchmark.sh 256 4 8 2 512 32 4

# Large model
bash benchmark.sh 1024 12 16 2 8192 128 16
```

## JSON Format

The generated JSON file has this structure:

```json
{
  "meta": {
    "B": 2,              // Batch size
    "S": 5,              // Sequence length
    "head_dim": 8        // Hidden dim per head
  },
  "config": {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "intermediate_size": 256,
    // ... other config fields
  },
  "inputs": {
    "input_ids": {
      "data": "base64_encoded_bytes",
      "shape": [2, 5],
      "dtype": "int32",
      "preview": [1, 5, 3, ...]  // First N values
    },
    "attention_mask": { /* similar */ }
  },
  "rope": {
    "cos": { /* rope cos cache */ },
    "sin": { /* rope sin cache */ }
  },
  "weights": {
    "tok_embedding": { /* (vocab, hidden) */ },
    "final_rms": { /* (hidden) */ },
    "lm_head": { /* (vocab, hidden) */ },
    "layers": [
      {
        "rms_attn": { /* (hidden) */ },
        "rms_ffn": { /* (hidden) */ },
        "wq": { /* (heads*D, hidden) */ },
        "wk": { /* (kvh*D, hidden) */ },
        "wv": { /* (kvh*D, hidden) */ },
        "wo": { /* (hidden, heads*D) */ },
        "w_gate": { /* (inter, hidden) */ },
        "w_up": { /* (inter, hidden) */ },
        "w_down": { /* (hidden, inter) */ }
      },
      // ... more layers
    ]
  }
}
```

## Configuration Parameters

### `generate_random_model.py`

| Param | Default | Description |
|-------|---------|-------------|
| `--hidden` | 64 | Model hidden size |
| `--layers` | 2 | Number of transformer layers |
| `--heads` | 8 | Number of attention heads |
| `--kvh` | 2 | Number of KV heads (for grouped query attention) |
| `--vocab` | 128 | Vocabulary size |
| `--max-pos` | 128 | Maximum position embeddings |
| `--seq-len` | 5 | Input sequence length |
| `--batch-size` | 2 | Batch size |
| `--seed` | 123 | Random seed for reproducibility |
| `--output` | `dump_minimind/minimind.json` | Output JSON path |

### Environment Variables for Forward Pass

```bash
JSON_PATH=dump_minimind/minimind.json    # Input JSON path
DUMP_DIR=dump_minimind                    # Output directory
WARMUP=5                                   # Number of warmup iterations
```

## Workflow Overview

```
generate_random_model.py
        ↓
   minimind.json
        ↓
    ┌───┴────┐
    ↓        ↓
torch_fwd  cpp_fwd
    ↓        ↓
logits_torch  logits_cpp
```

## Key Design Decisions

1. **Base64 Encoding**: Weights are stored as Base64-encoded binary data
   - Human-readable JSON structure
   - Efficient binary storage
   - Metadata (shape, dtype) included

2. **No Model Training**: Weights are random, not trained
   - Fast model generation
   - Useful for performance testing
   - Not suitable for correctness validation

3. **Separate JSON Loading Timing**: 
   - Python forward pass excludes JSON loading time
   - C++ loads JSON and times only forward pass
   - Enables accurate performance comparison

4. **Intermediate Size Auto-Calculation**:
   - Matches PyTorch's FeedForward calculation
   - `inter = hidden * 4`
   - `inter = 64 * ((inter + 63) // 64)` (aligned to 64)

## Troubleshooting

### JSON Loading Fails
- Ensure `minimind.json` exists in specified `JSON_PATH`
- Check JSON syntax: `python -m json.tool dump_minimind/minimind.json`

### Shape Mismatch Errors
- Verify config parameters (hidden, heads, vocab)
- Check intermediate_size matches expected value

### C++ Logits All Zeros
- Ensure forward pass is correctly implemented
- Check memory allocation for workspace buffers

## Performance Tips

1. **Disable Warmup for Minimal Latency**:
   ```bash
   export WARMUP=0
   ```

2. **Vary Input Sizes for Profiling**:
   ```bash
   # Small
   bash benchmark.sh 64 2 8 2 128 5 2
   
   # Medium
   bash benchmark.sh 512 6 8 2 4096 64 8
   
   # Large
   bash benchmark.sh 2048 24 32 4 32768 1024 16
   ```

3. **Compare PyTorch vs C++**:
   ```bash
   # Extract timing from logits files
   python -c "
   import numpy as np
   torch_logits = np.load('dump_minimind/logits_torch.npy')
   cpp_logits = np.load('dump_minimind/logits_cpp.npy')
   print('Logits match:', np.allclose(torch_logits, cpp_logits, rtol=1e-4))
   "
   ```

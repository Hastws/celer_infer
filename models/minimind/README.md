# MiniMind Model

This directory contains the MiniMind baseline micro implementation weights and configuration.

## Files

- `config.json` - Model configuration (hidden size, num layers, etc.)
- `minimind.json` - Model weights exported from PyTorch (Base64-encoded)
- `README.md` - This file

## Quick Start

```bash
# Export model weights
python -m python export --model minimind

# Verify consistency between PyTorch and C++
python -m python validate --model minimind

# Run C++ inference
./build/minimind models/minimind/minimind.json
```

## Model Architecture

- **Type**: Transformer with RoPE and RMSNorm
- **Hidden Size**: 64
- **Number of Layers**: 2
- **Attention Heads**: 8
- **FFN Intermediate**: 256
- **Vocabulary Size**: 128

See [../README.md](../README.md) for architecture details.

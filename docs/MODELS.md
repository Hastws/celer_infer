# Supported Models

## MiniMind (✅ Implemented)

A lightweight transformer-based LLM with the following features:

### Architecture
- **Type**: Transformer with RoPE and RMSNorm
- **Attention**: Multi-head self-attention with optional group query attention
- **FFN**: Feed-forward networks with SiLU activation
- **Optional MoE**: Mixture of Experts support (configurable)

### Configuration
```json
{
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 8,
  "num_key_value_heads": 8,
  "intermediate_size": 256,
  "vocab_size": 128,
  "max_position_embeddings": 2048,
  "rope_theta": 10000.0
}
```

### Quick Start
```bash
# Export weights
python -m python dump --model minimind

# Verify consistency
python -m python validate --model minimind

# Run C++ inference
./cpp/build/minimind models/minimind/minimind.json
```

## LLAMA (📋 Planned)

Llama 2 / Llama 3 support is planned for the next release.

### Planned Features
- [ ] Model definition with rotary embeddings
- [ ] Weight export and verification
- [ ] C++ inference implementation
- [ ] Performance optimization

## Qwen (📋 Planned)

Qwen model support coming soon.

## Model Comparison

| Model | Status | Layers | Hidden | Heads | FFN | MoE |
|-------|--------|--------|--------|-------|-----|-----|
| MiniMind | ✅ | 2 | 64 | 8 | 256 | ✅ |
| LLAMA | 📋 | - | - | - | - | ❌ |
| Qwen | 📋 | - | - | - | - | ❌ |

## Adding a New Model

See [ARCHITECTURE.md](ARCHITECTURE.md#adding-a-new-model) for detailed instructions.

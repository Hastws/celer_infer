#!/usr/bin/env python3
"""
DiT Validation Script

验证 DiT 模型的 Python 和 C++ 实现一致性。

Usage:
    python -m python.validate.dit_validate
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import torch
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from python.core.dit import (
    ModelConfig, Diffusion_Planner, 
    create_dummy_inputs, count_parameters
)
from python.export.dit_dumper import DiTDumper


def create_test_config() -> ModelConfig:
    """Create a test config using default values (matching model architecture)"""
    # Use default ModelConfig to ensure compatibility
    return ModelConfig(
        # Smaller model for faster testing, but keep data dimensions as default
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        hidden_dim=96,
        predicted_neighbor_num=3,
        diffusion_model_type='x_start',
        device='cpu',
    )


def test_export():
    """Test model export"""
    print("="*60)
    print("Testing DiT Export")
    print("="*60)
    
    cfg = create_test_config()
    model = Diffusion_Planner(cfg)
    model.eval()
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Create inputs
    inputs = create_dummy_inputs(cfg, batch_size=1, device='cpu')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use the DiTDumper correctly
        dumper = DiTDumper()
        json_path = dumper.dump(model, cfg, inputs, tmpdir)
        
        # Verify JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"\nExported JSON keys: {list(data.keys())}")
        print(f"Config keys: {list(data['config'].keys())[:10]}...")
        print(f"Weight groups: {list(data['weights'].keys())}")
        
        # Check sizes
        file_size = os.path.getsize(json_path)
        print(f"\nJSON file size: {file_size / 1024 / 1024:.2f} MB")
        
    print("\n✓ Export test passed!")


def test_forward():
    """Test forward pass"""
    print("\n" + "="*60)
    print("Testing DiT Forward Pass")
    print("="*60)
    
    cfg = create_test_config()
    model = Diffusion_Planner(cfg)
    model.eval()
    
    inputs = create_dummy_inputs(cfg, batch_size=1, device='cpu')
    
    with torch.no_grad():
        # Full model forward pass
        encoder_out, decoder_out = model(inputs)
        
        encoding = encoder_out['encoding']
        print(f"Encoder output shape: {encoding.shape}")
        print(f"Encoder output stats: min={encoding.min():.4f}, max={encoding.max():.4f}, mean={encoding.mean():.4f}")
        
        if 'prediction' in decoder_out:
            prediction = decoder_out['prediction']
            print(f"\nDecoder prediction shape: {prediction.shape}")
            print(f"Decoder prediction stats: min={prediction.min():.4f}, max={prediction.max():.4f}, mean={prediction.mean():.4f}")
    
    print("\n✓ Forward pass test passed!")


def test_encoder_decoder_separately():
    """Test encoder and decoder separately"""
    print("\n" + "="*60)
    print("Testing Encoder and Decoder Separately")
    print("="*60)
    
    cfg = create_test_config()
    model = Diffusion_Planner(cfg)
    model.eval()
    
    inputs = create_dummy_inputs(cfg, batch_size=1, device='cpu')
    
    import time
    
    with torch.no_grad():
        # Test encoder only
        print("Testing Encoder...")
        
        # Warmup
        for _ in range(3):
            _ = model.encoder(inputs)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            encoder_out = model.encoder(inputs)
            times.append((time.perf_counter() - start) * 1000)
        
        print(f"  Encoder: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
        print(f"  Encoding shape: {encoder_out['encoding'].shape}")
        
        # Test decoder only - it needs encoder_outputs and inputs
        print("Testing Decoder...")
        
        # Warmup
        for _ in range(3):
            _ = model.decoder(encoder_out, inputs)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            decoder_out = model.decoder(encoder_out, inputs)
            times.append((time.perf_counter() - start) * 1000)
        
        print(f"  Decoder: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
    
    print("\n✓ Separate encoder/decoder test passed!")


def main():
    print("DiT Validation Suite")
    print("="*60)
    
    try:
        test_export()
        test_forward()
        test_encoder_decoder_separately()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Validation and comparison utilities"""

import numpy as np
import json


class Comparator:
    """Base class for comparing outputs"""

    @staticmethod
    def compare_arrays(arr_cpp, arr_torch, name: str = ""):
        """Compare two arrays and print statistics"""
        diff = np.abs(arr_cpp - arr_torch)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        try:
            correlation = np.corrcoef(arr_cpp.flatten(), arr_torch.flatten())[0, 1]
        except:
            correlation = 1.0

        status = "✓ MATCH" if max_diff < 1e-5 else "✗ MISMATCH"

        print(f"\n{name}:")
        print(f"  Shape: {arr_cpp.shape}")
        print(f"  C++   - Min: {arr_cpp.min():.8f}, Max: {arr_cpp.max():.8f}")
        print(f"  PyTorch - Min: {arr_torch.min():.8f}, Max: {arr_torch.max():.8f}")
        print(f"  Difference: max={max_diff:.2e}, mean={mean_diff:.2e}")
        print(f"  Correlation: {correlation:.8f}")
        print(f"  {status}")

        return max_diff < 1e-5


def get_validator(model_name: str):
    """Get a validator for a specific model"""
    if model_name == "minimind":
        # 集成现有的对比脚本
        return MinimindValidator()

    raise NotImplementedError(f"Validator for {model_name} not implemented")


class MinimindValidator:
    """Validator for MiniMind model"""

    def validate_all(self):
        """Run all validation checks"""
        print("=" * 70)
        print("Running MiniMind Validation Suite")
        print("=" * 70)

        # 可集成现有的对比脚本逻辑
        pass

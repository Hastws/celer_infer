"""Inference verification utilities"""

import json
import os


def get_verifier(model_name: str):
    """
    Get a verifier for a specific model.

    Args:
        model_name: Name of the model (e.g., 'minimind')

    Returns:
        Verifier instance
    """
    if model_name == "minimind":
        from .minimind_forward import MinimindVerifier
        return MinimindVerifier()

    raise NotImplementedError(f"Verifier for {model_name} not implemented")


def verify_consistency(model_name: str, config_path: str = None):
    """
    Verify consistency between PyTorch and C++ inference.

    Args:
        model_name: Name of the model
        config_path: Optional path to custom config file
    """
    if config_path is None:
        config_path = f"models/{model_name}/config.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    verifier = get_verifier(model_name)
    return verifier.verify(config_path)

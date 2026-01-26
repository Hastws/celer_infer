"""Core model definitions"""

import json
import os
from pathlib import Path

# Model registry - 支持多模型
_MODEL_REGISTRY = {
    "minimind": {
        "model_class": "minimind_model.MiniMindForCausalLM",
        "config": "models/minimind/config.json",
    }
    # 未来可添加其他模型
    # "llama": {...},
}


def get_model(model_name: str, config_path: str = None):
    """
    Get a model instance by name.

    Args:
        model_name: Name of the model (e.g., 'minimind', 'llama')
        config_path: Optional path to custom config file

    Returns:
        Model instance
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODEL_REGISTRY.keys())}")

    model_info = _MODEL_REGISTRY[model_name]

    # Load config
    if config_path is None:
        config_path = model_info["config"]

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_full = json.load(f)

    # Extract actual model config (handle both nested and flat formats)
    if "config" in config_full:
        config_dict = config_full["config"]
    else:
        config_dict = config_full

    # Import and instantiate model
    if model_name == "minimind":
        from .minimind_model import MiniMindConfig, MiniMindForCausalLM
        cfg = MiniMindConfig(**config_dict)
        return MiniMindForCausalLM(cfg)

    raise NotImplementedError(f"Model {model_name} not yet implemented")


def register_model(name: str, model_class: str, config_path: str):
    """Register a new model."""
    _MODEL_REGISTRY[name] = {
        "model_class": model_class,
        "config": config_path,
    }


def list_models():
    """List all available models."""
    return list(_MODEL_REGISTRY.keys())

"""Debug utilities"""

import json
import os


def get_debugger(model_name: str):
    """
    Get a debugger for a specific model.

    Args:
        model_name: Name of the model (e.g., 'minimind')

    Returns:
        Debugger instance
    """
    if model_name == "minimind":
        # 可在此处导入 MiniMind 特定的调试工具
        from .minimind_debug import MinimindDebugger
        return MinimindDebugger()

    raise NotImplementedError(f"Debugger for {model_name} not implemented")


def extract_layer_output(model_name: str, layer_id: int):
    """Extract output from a specific layer for debugging."""
    debugger = get_debugger(model_name)
    return debugger.extract_layer(layer_id)

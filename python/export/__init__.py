"""Export utilities for dumping model weights"""

import json
import os


def get_dumper(model_name: str):
    """
    Get a dumper for a specific model.

    Args:
        model_name: Name of the model (e.g., 'minimind')

    Returns:
        Dumper instance
    """
    if model_name == "minimind":
        from .minimind_dumper import MinimindDumper
        return MinimindDumper()

    raise NotImplementedError(f"Dumper for {model_name} not implemented")


def dump_model(model_name: str, model, output_dir: str = None):
    """
    Dump model weights to JSON.

    Args:
        model_name: Name of the model
        model: Model instance
        output_dir: Output directory (defaults to models/{model_name})
    """
    if output_dir is None:
        output_dir = f"models/{model_name}"

    os.makedirs(output_dir, exist_ok=True)

    dumper = get_dumper(model_name)
    return dumper.dump(model, output_dir)

"""Validation helper - full consistency validation workflow"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import get_model
from export.minimind_dumper import MinimindDumper
from inference.minimind_forward import MinimindVerifier


def validate_model(model_name: str = "minimind") -> bool:
    """
    Run full validation pipeline for a model.

    Steps:
    1. Dump model weights to JSON
    2. Verify C++ inference against PyTorch reference

    Args:
        model_name: Name of model to validate

    Returns:
        True if validation successful
    """
    print(f"\n[INFO] Starting validation for model: {model_name}")

    try:
        # Step 1: Get and dump model
        print(f"\n[STEP 1] Dumping model weights...")
        model = get_model(model_name)
        dumper = MinimindDumper()
        
        # Get output directory
        import os
        from pathlib import Path
        root_dir = Path(__file__).parent.parent.parent
        model_dir = root_dir / "models" / model_name
        output_dir = str(model_dir)
        
        json_path = dumper.dump(model, output_dir=output_dir)
        print(f"[OK] Model dumped to: {json_path}")

        # Step 2: Verify consistency
        print(f"\n[STEP 2] Verifying consistency with PyTorch...")
        verifier = MinimindVerifier()
        success = verifier.verify(json_path)

        if success:
            print(f"\n[SUCCESS] Model validation complete!")
            return True
        else:
            print(f"\n[ERROR] Consistency check failed!")
            return False

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate model consistency")
    parser.add_argument(
        "--model",
        type=str,
        default="minimind",
        help="Model to validate (default: minimind)",
    )

    args = parser.parse_args()
    success = validate_model(args.model)
    sys.exit(0 if success else 1)

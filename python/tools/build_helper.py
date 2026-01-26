"""Build helper - wrapper for C++ compilation"""

import os
import subprocess
import sys
from pathlib import Path


def build_cpp(build_dir: str = "build") -> bool:
    """
    Build the C++ inference engine.

    Args:
        build_dir: Directory for build artifacts (default: build)

    Returns:
        True if successful
    """
    # Get paths relative to workspace root
    root_dir = Path(__file__).parent.parent.parent
    cpp_dir = root_dir / "cpp"
    
    # Create build directory in cpp directory
    build_path = cpp_dir / build_dir
    build_path.mkdir(parents=True, exist_ok=True)

    try:
        # Run cmake from build directory, pointing to cpp directory
        print(f"[INFO] Running cmake in {build_path}...")
        subprocess.run(
            ["cmake", ".."],
            cwd=str(build_path),
            check=True,
        )

        # Run make
        print("[INFO] Running make...")
        subprocess.run(
            ["make"],
            cwd=str(build_path),
            check=True,
        )

        print("[OK] C++ build successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e}")
        return False


if __name__ == "__main__":
    success = build_cpp()
    sys.exit(0 if success else 1)

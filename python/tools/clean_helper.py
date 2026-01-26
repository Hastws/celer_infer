"""Clean helper - cleanup build artifacts and caches"""

import os
import shutil
import sys
from pathlib import Path


def clean_build_artifacts(build_dirs: list[str] = None) -> bool:
    """
    Clean C++ build artifacts.

    Args:
        build_dirs: List of directories to clean (default: [build, cpp/build])

    Returns:
        True if successful
    """
    if build_dirs is None:
        build_dirs = ["build", "cpp/build"]

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    try:
        for build_dir in build_dirs:
            full_path = os.path.join(root_dir, build_dir)
            if os.path.exists(full_path):
                print(f"[INFO] Removing {full_path}")
                shutil.rmtree(full_path)
                print(f"[OK] Cleaned {build_dir}")
            else:
                print(f"[INFO] {build_dir} not found (skipping)")

        return True
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        return False


def clean_python_cache() -> bool:
    """
    Clean Python cache files (__pycache__, .pyc, etc).

    Returns:
        True if successful
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    try:
        # Remove __pycache__ directories
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if "__pycache__" in dirnames:
                cache_path = os.path.join(dirpath, "__pycache__")
                print(f"[INFO] Removing {cache_path}")
                shutil.rmtree(cache_path)
                print(f"[OK] Cleaned __pycache__ in {dirpath}")

        return True
    except Exception as e:
        print(f"[ERROR] Cache cleanup failed: {e}")
        return False


def clean_all(build_only: bool = False) -> bool:
    """
    Clean all artifacts.

    Args:
        build_only: If True, only clean C++ build (not Python cache)

    Returns:
        True if successful
    """
    print("\n[INFO] Starting cleanup...")

    success = True

    # Clean C++ builds
    if not clean_build_artifacts():
        success = False

    # Clean Python cache
    if not build_only:
        if not clean_python_cache():
            success = False

    if success:
        print("\n[SUCCESS] Cleanup complete!")
    else:
        print("\n[ERROR] Cleanup had errors!")

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean build artifacts and caches")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only clean C++ build artifacts (not Python cache)",
    )

    args = parser.parse_args()
    success = clean_all(build_only=args.build_only)
    sys.exit(0 if success else 1)

"""Benchmark helper - performance benchmarking utilities"""

import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.minimind_forward import MinimindVerifier


def benchmark_model(model_name: str = "minimind", num_iterations: int = 10) -> bool:
    """
    Benchmark model inference performance.

    Args:
        model_name: Name of model to benchmark
        num_iterations: Number of iterations for warmup + measurement

    Returns:
        True if successful
    """
    print(f"\n[INFO] Benchmarking {model_name} inference ({num_iterations} iterations)...")

    try:
        verifier = MinimindVerifier()

        # Get JSON path for the model
        import os
        root_dir = Path(__file__).parent.parent.parent
        model_dir = root_dir / "models" / model_name
        json_path = str(model_dir / f"{model_name}.json")

        if not Path(json_path).exists():
            print(f"[ERROR] Model JSON not found at {json_path}")
            print(f"[INFO] Running dump first...")
            from export.minimind_dumper import MinimindDumper
            dumper = MinimindDumper()
            json_path = dumper.dump(model_name)

        # Warmup + measurement
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            result = verifier.verify(json_path, skip_comparison=True)
            elapsed = time.time() - start_time
            times.append(elapsed)

            if i == 0:
                print(f"  [Warmup] {elapsed:.4f}s")
            else:
                print(f"  [Iter {i}] {elapsed:.4f}s")

            if not result:
                print(f"[ERROR] Inference failed at iteration {i}")
                return False

        # Statistics
        avg_time = sum(times[1:]) / len(times[1:])  # Exclude warmup
        min_time = min(times[1:])
        max_time = max(times[1:])

        print(f"\n[STATS] (excluding warmup):")
        print(f"  Average: {avg_time:.4f}s")
        print(f"  Min: {min_time:.4f}s")
        print(f"  Max: {max_time:.4f}s")
        print(f"  Throughput: {1.0/avg_time:.2f} inference/sec")

        return True

    except Exception as e:
        print(f"[ERROR] Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark model inference")
    parser.add_argument(
        "--model",
        type=str,
        default="minimind",
        help="Model to benchmark (default: minimind)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations (default: 10)",
    )

    args = parser.parse_args()
    success = benchmark_model(args.model, args.iterations)
    sys.exit(0 if success else 1)

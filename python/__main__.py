"""
CelerInfer - Unified Command Line Interface

Usage:
    python -m python dump [--model MODEL] [--output DIR]
    python -m python validate [--model MODEL]
    python -m python debug [--model MODEL] [--layer LAYER]
    python -m python build [--build-dir DIR]
    python -m python run-validation [--model MODEL]
    python -m python clean [--build-only]
    python -m python benchmark [--model MODEL] [--iterations N]
    python -m python list-models
"""

import argparse
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_dump(args):
    """Export model weights to JSON"""
    from python.core import get_model
    from python.export import dump_model

    model_name = args.model or "minimind"
    output_dir = args.output or f"models/{model_name}"

    print(f"[INFO] Dumping {model_name} model to {output_dir}")
    try:
        # Instantiate the model first
        model = get_model(model_name)
        # Then dump it
        dump_model(model_name, model, output_dir=output_dir)
        print(f"[OK] Model dumped successfully to {output_dir}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_validate(args):
    """Verify consistency between PyTorch and C++ inference"""
    from python.inference import verify_consistency

    model_name = args.model or "minimind"

    print(f"[INFO] Validating {model_name} model")
    try:
        verify_consistency(model_name)
        print(f"[OK] Validation passed")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_debug(args):
    """Run debugging tools"""
    from python.debug import get_debugger

    model_name = args.model or "minimind"

    print(f"[INFO] Running debugger for {model_name}")
    try:
        debugger = get_debugger(model_name)
        if args.layer is not None:
            debugger.extract_layer(args.layer)
        else:
            debugger.debug_all()
        print(f"[OK] Debugging complete")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_list_models(args):
    """List available models"""
    from python.core import list_models

    models = list_models()
    print(f"Available models: {', '.join(models)}")


def cmd_build(args):
    """Build C++ inference engine"""
    from python.tools import build_cpp

    build_dir = args.build_dir or "build"
    print(f"[INFO] Building C++ engine (build dir: {build_dir})")
    try:
        success = build_cpp(build_dir)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_run_validation(args):
    """Run full validation pipeline"""
    from python.tools import validate_model

    model_name = args.model or "minimind"
    print(f"[INFO] Running full validation for {model_name}")
    try:
        success = validate_model(model_name)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_clean(args):
    """Clean build artifacts and caches"""
    from python.tools import clean_all

    build_only = args.build_only or False
    print(f"[INFO] Cleaning artifacts (build_only: {build_only})")
    try:
        success = clean_all(build_only=build_only)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_benchmark(args):
    """Benchmark model inference"""
    from python.tools import benchmark_model

    model_name = args.model or "minimind"
    iterations = args.iterations or 10
    print(f"[INFO] Benchmarking {model_name} ({iterations} iterations)")
    try:
        success = benchmark_model(model_name, iterations)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)



def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="CelerInfer - LLM Inference Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # dump command
    dump_parser = subparsers.add_parser("dump", help="Export model weights")
    dump_parser.add_argument("--model", default="minimind", help="Model name")
    dump_parser.add_argument("--output", help="Output directory")
    dump_parser.set_defaults(func=cmd_dump)

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Verify PyTorch↔C++ consistency")
    validate_parser.add_argument("--model", default="minimind", help="Model name")
    validate_parser.set_defaults(func=cmd_validate)

    # debug command
    debug_parser = subparsers.add_parser("debug", help="Run debugging tools")
    debug_parser.add_argument("--model", default="minimind", help="Model name")
    debug_parser.add_argument("--layer", type=int, help="Extract specific layer")
    debug_parser.set_defaults(func=cmd_debug)

    # list-models command
    subparsers.add_parser("list-models", help="List available models").set_defaults(
        func=cmd_list_models
    )

    # build command
    build_parser = subparsers.add_parser("build", help="Build C++ inference engine")
    build_parser.add_argument("--build-dir", default="build", help="Build directory")
    build_parser.set_defaults(func=cmd_build)

    # run-validation command
    run_validation_parser = subparsers.add_parser(
        "run-validation", help="Run full validation pipeline"
    )
    run_validation_parser.add_argument("--model", default="minimind", help="Model name")
    run_validation_parser.set_defaults(func=cmd_run_validation)

    # clean command
    clean_parser = subparsers.add_parser("clean", help="Clean build artifacts")
    clean_parser.add_argument(
        "--build-only", action="store_true", help="Only clean C++ builds"
    )
    clean_parser.set_defaults(func=cmd_clean)

    # benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark inference")
    benchmark_parser.add_argument("--model", default="minimind", help="Model name")
    benchmark_parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations"
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)


    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

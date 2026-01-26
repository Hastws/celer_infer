#!/bin/bash
# Build C++ inference engine

set -e

cd "$(dirname "$0")"/..

echo "========================================"
echo "Building CelerInfer C++ Inference Engine"
echo "========================================"

# Create build directory
mkdir -p cpp/build

# Build
cd cpp/build
cmake ..
make -j$(nproc)

echo ""
echo "✅ Build complete!"
echo "Executable: ./build/minimind"

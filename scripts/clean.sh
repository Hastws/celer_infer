#!/bin/bash
# Clean build artifacts

echo "Cleaning build artifacts..."

# C++ build
rm -rf cpp/build
mkdir -p cpp/build

# Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

# Build directory
rm -rf build

echo "✅ Clean complete!"

#!/bin/bash
# cleanup_legacy.sh - Remove deprecated script/ directory (optional)
#
# The script/ directory contains old development files that have been
# migrated to the python/ directory. This script safely removes them.
#
# Usage: bash cleanup_legacy.sh

set -e

echo "========================================"
echo "CelerInfer Legacy Script Cleanup"
echo "========================================"
echo ""

SCRIPT_DIR="script"
ARCHIVE_DIR=".archive/legacy"

# Check if script directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "✅ '$SCRIPT_DIR' directory not found - nothing to clean"
    exit 0
fi

echo "⚠️  This will archive/remove the legacy '$SCRIPT_DIR/' directory"
echo ""
echo "Files to be archived:"
ls -1 "$SCRIPT_DIR"/ | sed 's/^/  - /'
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Archiving legacy files..."

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

# Move script directory
if [ -d "$SCRIPT_DIR" ]; then
    mv "$SCRIPT_DIR" "$ARCHIVE_DIR/script_old_$(date +%Y%m%d_%H%M%S)"
    echo "✅ Legacy files archived to: $ARCHIVE_DIR/"
fi

echo ""
echo "🎉 Cleanup complete!"
echo ""
echo "New imports:"
echo "  from python.core.minimind_model import MiniMindForCausalLM"
echo "  from python.inference.minimind_forward import MinimindVerifier"
echo "  from python.tools import build_cpp, validate_model, benchmark_model"
echo ""
echo "New CLI commands:"
echo "  python -m python build"
echo "  python -m python run-validation"
echo "  python -m python clean"
echo "  python -m python benchmark"

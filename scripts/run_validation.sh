#!/bin/bash
# Run complete validation pipeline

set -e

MODEL="${1:-minimind}"
echo "========================================"
echo "Validating $MODEL Model"
echo "========================================"

# Export model weights
echo ""
echo "Step 1: Dumping model weights..."
python -m python dump --model "$MODEL"

# Run PyTorch inference
echo ""
echo "Step 2: Running PyTorch inference..."
python python/inference/minimind_forward.py

# Run C++ inference
echo ""
echo "Step 3: Running C++ inference..."
if [ ! -f "cpp/build/minimind" ]; then
    echo "Building C++ first..."
    bash scripts/build_cpp.sh
fi

./cpp/build/minimind models/"$MODEL"/minimind.json

# Run validation
echo ""
echo "Step 4: Running validation..."
python python/validate/compare_logits.py

echo ""
echo "✅ Validation complete!"

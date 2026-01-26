#!/bin/bash
# Example usage script for random model generation and benchmarking

set -e

echo "=========================================="
echo "CelerInfer Random Model Benchmarking"
echo "=========================================="
echo ""

# Configuration
HIDDEN=${1:-64}
LAYERS=${2:-2}
HEADS=${3:-8}
KVH=${4:-2}
VOCAB=${5:-128}
SEQ_LEN=${6:-5}
BATCH_SIZE=${7:-2}
SEED=${8:-123}

OUTPUT_DIR="dump_minimind"
JSON_PATH="${OUTPUT_DIR}/minimind.json"

echo "Parameters:"
echo "  Hidden size: $HIDDEN"
echo "  Layers: $LAYERS"
echo "  Heads: $HEADS"
echo "  KV Heads: $KVH"
echo "  Vocab: $VOCAB"
echo "  Seq len: $SEQ_LEN"
echo "  Batch size: $BATCH_SIZE"
echo "  Seed: $SEED"
echo ""

# Step 1: Generate random model
echo "Step 1: Generating random model JSON..."
python script/generate_random_model.py \
    --hidden "$HIDDEN" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --kvh "$KVH" \
    --vocab "$VOCAB" \
    --seq-len "$SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --output "$JSON_PATH"

echo ""

# Step 2: Run PyTorch forward pass
echo "Step 2: Running PyTorch forward pass..."
export JSON_PATH="$JSON_PATH"
export DUMP_DIR="$OUTPUT_DIR"
export WARMUP=5

python script/llm_minimind_forward.py

echo ""

# Step 3: Build and run C++ inference
echo "Step 3: Building and running C++ inference..."
cd cpp
mkdir -p build && cd build
cmake .. > /dev/null 2>&1
make > /dev/null 2>&1
cd ..

echo "Running C++ inference..."
./build/base_line_micro "../${JSON_PATH}" "../${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "Benchmarking complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - ${JSON_PATH}: Model weights (JSON format)"
echo "  - ${OUTPUT_DIR}/logits_torch.npy: PyTorch logits"
echo "  - ${OUTPUT_DIR}/logits_cpp.npy: C++ logits"

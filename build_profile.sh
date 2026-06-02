#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build_release}"
TRAIN_DIR="${TRAIN_DIR:-profile_training_data}"
OUTPUT_MODEL="${OUTPUT_MODEL:-/tmp/rllm-profile-small.json}"
LAYERS="${LAYERS:-1}"
EPOCHS="${EPOCHS:-1}"

EPOCHS=10

if [[ ! -d "$TRAIN_DIR" ]]; then
    echo "Training directory '$TRAIN_DIR' does not exist. Set TRAIN_DIR to an existing folder."
    exit 1
fi

echo "Configuring profiling build in '$BUILD_DIR'..."
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARALLEL_BACKEND=fastfork \
    -DOFFLOAD_BACKEND=vulkan

echo "Building profiling target rllm_prof..."
cmake --build "$BUILD_DIR" --parallel --target rllm_prof

echo "Running small profiling training batch..."
/usr/bin/time -f 'ELAPSED=%e' \
    "./$BUILD_DIR/rllm_prof" \
    --train \
    --train-dir "$TRAIN_DIR" \
    --layers "$LAYERS" \
    --epochs "$EPOCHS" \
    --checkpoint-interval 0 \
    -o "$OUTPUT_MODEL"
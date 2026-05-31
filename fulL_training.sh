#!/usr/bin/env bash
set -euo pipefail

STAGE0_MODEL="staged/stage0_model.json"
STAGE1_MODEL="staged/state1_model.json"

echo "Configuring and building release before staged training..."
sh ./build_release.sh

mkdir -p models
mkdir -p staged

echo "--- Stage 0: training on training_data0 ---"
./build_release/rllm --train \
    --train-dir training_data0 \
    -o "$STAGE0_MODEL" \
    --method random_line_random_len \
    --epochs 50 \
    --checkpoint-interval 30

echo "Normalizing training_data1 with training_postprocessor.py..."
python3 ./training_postprocessor.py --dir training_data1

echo "--- Stage 1: training on training_data1 from $STAGE0_MODEL ---"
./build_release/rllm --train \
    --train-dir training_data1 \
    -i "$STAGE0_MODEL" \
    -o "$STAGE1_MODEL" \
    --method increasingly_longer \
    --epochs 20 \
    --layers 4 \
    --checkpoint-interval 30

echo "Finished staged training."
echo "Stage 0 model: $STAGE0_MODEL"
echo "Stage 1 model: $STAGE1_MODEL"
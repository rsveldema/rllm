#!/usr/bin/env bash
set -euo pipefail

echo "Configuring and building release before training..."
sh ./build_release.sh

TRAIN_DIR="${TRAIN_DIR:-training_data1}"

echo "Locating checkpoint to resume from..."

echo "Normalizing training_data1 with training_postprocessor.py..."
python3 ./training_postprocessor.py --dir training_data1

echo "Formatting training_data0/*.cpp with maximum line length..."
if compgen -G "training_data0/*.cpp" > /dev/null; then
    clang-format -i --style='{BasedOnStyle: LLVM, ColumnLimit: 0}' training_data0/*.cpp
else
    echo "No .cpp files found in training_data0"
fi

if [ ! -d "$TRAIN_DIR" ]; then
    echo "Training directory '$TRAIN_DIR' does not exist. Set TRAIN_DIR to an existing folder."
    exit 1
fi

# Resume from an explicit model path, then after_training.st, then latest checkpoint.
# Use RESUME_MODEL=/path/to/model.st to override.
if [ "${FRESH_START:-0}" != "0" ]; then
    echo "FRESH_START=${FRESH_START}: ignoring existing checkpoints and starting from random weights."
    input_arg=""
elif [ -n "${RESUME_MODEL:-}" ] && [ -f "${RESUME_MODEL}" ]; then
    echo "Resuming from ${RESUME_MODEL}"
    input_arg="-i ${RESUME_MODEL}"
elif [ -f "models/after_training.st" ]; then
    echo "Resuming from models/after_training.st"
    input_arg="-i models/after_training.st"
else
    shopt -s nullglob
    latest_checkpoint=""
    for checkpoint in models/checkpoint-*.st; do
        if [[ -z "$latest_checkpoint" || "$checkpoint" -nt "$latest_checkpoint" ]]; then
            latest_checkpoint="$checkpoint"
        fi
    done
    shopt -u nullglob

    if [ -n "$latest_checkpoint" ]; then
        echo "Resuming from $latest_checkpoint"
        input_arg="-i $latest_checkpoint"
    else
        echo "No checkpoint found, starting from random weights."
        input_arg=""
    fi
fi

echo "--- Starting training ---"

export NAN_FINDING_MODE=1


./build_release/rllm --train $input_arg \
    -o models/after_training.st \
    --train-dir "$TRAIN_DIR" \
     --filter iuring \
     --filter simple \
     --filter self \
     --filter preprocessing \
     --method random_line_random_len \
     --epochs 20 \
     --layers 4 \
     --checkpoint-interval 30


# training Options:
#     --method random_line_random_len \
#     --method window:32 --epochs 20
#     --filter simple --method increasingly_longer --epochs 50
#     --method increasingly_longer

# ./build/rllm --train -i models/start.st \
#     -o models/after_training.st \
#      --filter simple --method three_tok --epochs 50
#

#./build/rllm --train -i models/start.st \
#    -o models/after_training.st \
#     --filter simple --method increasingly_longer --epochs 50

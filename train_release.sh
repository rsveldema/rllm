#!/usr/bin/env bash
set -euo pipefail

echo "Configuring and building release before training..."
sh ./build_release.sh

#TRAIN_DIR="${TRAIN_DIR:-training_data0}"
TRAIN_DIR="${TRAIN_DIR:-training_data1}"
RUNTIME_TOKENIZER_HEADER="build_release/generated/tokenizer_map.hpp"

runtime_vocab_size() {
    python3 ./runtime_vocab_size.py "$RUNTIME_TOKENIZER_HEADER"
}

model_vocab_size() {
    python3 ./model_vocab_size.py "$1"
}

resume_arg_for_model() {
    local candidate="$1"
    local required="${2:-0}"
    local runtime_vocab
    local model_vocab

    runtime_vocab="$(runtime_vocab_size)"
    model_vocab="$(model_vocab_size "$candidate")"

    if [[ "$model_vocab" == ERROR:* ]]; then
        echo "Cannot inspect resume model '$candidate': ${model_vocab#ERROR:}" >&2
        if [ "$required" != "0" ]; then
            exit 1
        fi
        return 1
    fi

    if [[ -n "$model_vocab" && "$model_vocab" != "$runtime_vocab" ]]; then
        echo "Skipping incompatible resume model '$candidate' (model vocab=$model_vocab, runtime vocab=$runtime_vocab)." >&2
        if [ "$required" != "0" ]; then
            exit 1
        fi
        return 1
    fi

    echo "-i $candidate"
}

delete_superseded_checkpoints() {
    local checkpoint

    shopt -s nullglob
    for checkpoint in models/checkpoint-[0-9]*.st; do
        echo "Deleting superseded checkpoint $checkpoint"
        rm -f -- "$checkpoint"
    done
    shopt -u nullglob
}

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

# Resume from an explicit model path, then the best window checkpoint, then the
# completed-training model, and finally the latest timed checkpoint.
# Use RESUME_MODEL=/path/to/model.st to override.
if [ "${FRESH_START:-0}" != "0" ]; then
    echo "FRESH_START=${FRESH_START}: ignoring existing checkpoints and starting from random weights."
    input_arg=""
elif [ -n "${RESUME_MODEL:-}" ] && [ -f "${RESUME_MODEL}" ]; then
    echo "Resuming from ${RESUME_MODEL}"
    input_arg="$(resume_arg_for_model "${RESUME_MODEL}" 1)"
elif [ -f "models/checkpoint-best-window.st" ]; then
    if input_arg="$(resume_arg_for_model "models/checkpoint-best-window.st")"; then
        echo "Resuming from models/checkpoint-best-window.st"
        delete_superseded_checkpoints
    else
        input_arg=""
    fi
elif [ -f "models/after_training.st" ]; then
    if input_arg="$(resume_arg_for_model "models/after_training.st")"; then
        echo "Resuming from models/after_training.st"
    else
        input_arg=""
    fi
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
        if input_arg="$(resume_arg_for_model "$latest_checkpoint")"; then
            echo "Resuming from $latest_checkpoint"
        else
            echo "No compatible checkpoint found, starting from random weights."
            input_arg=""
        fi
    else
        echo "No checkpoint found, starting from random weights."
        input_arg=""
    fi
fi

echo "--- Starting training ---"

#gdb --args  

#     --filter iuring \
#     --filter self \
#     --filter preprocessor \
#     --filter effective \
#     --filter modern \
#     --filter esched \
#     --filter lama \
#     --filter liburing \
#     --learning-rate-schedule simulated_annealing \


./build_release/rllm --train $input_arg \
    -o models/after_training.st \
    --train-dir "$TRAIN_DIR" \
     --filter simple \
     --method window:32 \
     --window-stride 1 \
     --epochs 80 \
     --disable-early-stopping \
     --disable-example-convergence \
     --layers 8 \
     --checkpoint-interval 300 \
     --learn-depth 4 \
     --learning-rate 0.00001 \
     --layer-learning-rate-multiplier 1.05 \
     --weight-initializer xavier-input-projections \
     --ffn-initializer xavier-input-projections \
     --embedding-initializer legacy-uniform \
     --learning-rate-schedule lowering \
     --simulated-annealing-initial-multiplier 9 \
     --simulated-annealing-decay-factor 0.7 \
     --simulated-annealing-decay-epochs 1 \
     --simulated-annealing-min-multiplier 0.02 \
     --micro-batch-size 256 \
     --vulkan-device R9700


#     --vulkan-device LLVMPIPE

#     --epoch-size 32 \

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

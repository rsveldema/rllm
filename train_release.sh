#!/usr/bin/env bash
set -euo pipefail

echo "Configuring and building release before training..."
sh ./build_release.sh

TRAIN_DIR="${TRAIN_DIR:-training_data1}"
RUNTIME_TOKENIZER_HEADER="build_release/generated/tokenizer_map.hpp"

runtime_vocab_size() {
    python3 - "$RUNTIME_TOKENIZER_HEADER" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"\bMAX\s*=\s*(\d+)", text)
if not match:
    raise SystemExit(f"Could not find TokenID::MAX in {sys.argv[1]}")
print(match.group(1))
PY
}

model_vocab_size() {
    python3 - "$1" <<'PY'
import json
import pathlib
import struct
import sys

path = pathlib.Path(sys.argv[1])
try:
    if path.suffix in {".st", ".safetensors"}:
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        print(header.get("__metadata__", {}).get("tokenizer_vocab_size", ""))
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            header = json.load(f)
        print(header.get("tokenizer_vocab_size", ""))
    else:
        print("")
except Exception as exc:
    print(f"ERROR:{exc}")
PY
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
    input_arg="$(resume_arg_for_model "${RESUME_MODEL}" 1)"
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

./build_release/rllm --train $input_arg \
    -o models/after_training.st \
    --train-dir "$TRAIN_DIR" \
     --filter iuring \
     --filter simple \
     --filter self \
     --filter preprocessor \
     --filter effective \
     --filter modern \
     --method random_line_random_len \
     --epochs 20 \
     --layers 3 \
     --checkpoint-interval 120 \
     --learn-depth 100 \
     --learning-rate 0.03 \
     --micro-batch-size 16 \
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

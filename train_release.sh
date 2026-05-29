
#!/usr/bin/env bash
set -euo pipefail

echo "Configuring and building release before training..."
cmake -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --parallel

echo "Locating checkpoint to resume from..."

# Resume from an explicit model path, then after_training.json, then latest checkpoint.
# Use RESUME_MODEL=/path/to/model.json to override.
if [ -n "${RESUME_MODEL:-}" ] && [ -f "${RESUME_MODEL}" ]; then
    echo "Resuming from ${RESUME_MODEL}"
    input_arg="-i ${RESUME_MODEL}"
elif [ -f "models/after_training.json" ]; then
    echo "Resuming from models/after_training.json"
    input_arg="-i models/after_training.json"
else
    shopt -s nullglob
    latest_checkpoint=""
    for checkpoint in models/checkpoint-*.json; do
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

./build_release/rllm --train $input_arg \
    -o models/after_training.json \
     --filter iuring \
     --filter simple \
     --filter self \
     --filter preprocessing \
     --method random_line_random_len \
     --epochs 20 \
     --checkpoint-interval 30




#     --method window:32 --epochs 20
#     --filter simple --method increasingly_longer --epochs 50
#     --method increasingly_longer

# ./build/rllm --train -i models/start.json \
#     -o models/after_training.json \
#      --filter simple --method three_tok --epochs 50
#

#./build/rllm --train -i models/start.json \
#    -o models/after_training.json \
#     --filter simple --method increasingly_longer --epochs 50
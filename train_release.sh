
# Resume from an explicit model path, then after_training.json, then latest checkpoint.
# Use RESUME_MODEL=/path/to/model.json to override.
if [ -n "${RESUME_MODEL:-}" ] && [ -f "${RESUME_MODEL}" ]; then
    echo "Resuming from ${RESUME_MODEL}"
    input_arg="-i ${RESUME_MODEL}"
elif [ -f "models/after_training.json" ]; then
    echo "Resuming from models/after_training.json"
    input_arg="-i models/after_training.json"
else
    latest_checkpoint=$(ls -t models/checkpoint-*.json 2>/dev/null | head -1)
    if [ -n "$latest_checkpoint" ]; then
        echo "Resuming from $latest_checkpoint"
        input_arg="-i $latest_checkpoint"
    else
        echo "No checkpoint found, starting from random weights."
        input_arg=""
    fi
fi

./build_release/rllm --train $input_arg \
    -o models/after_training.json \
     --filter guaranteed \
     --method random_line_random_len \
     --epochs 100 \
     --checkpoint-interval 10




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
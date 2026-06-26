
# Resume from the latest checkpoint if one exists, otherwise start fresh.
latest_checkpoint=$(ls -t models/checkpoint-*.json 2>/dev/null | head -1)
if [ -n "$latest_checkpoint" ]; then
    echo "Resuming from $latest_checkpoint"
    input_arg="-i $latest_checkpoint"
else
    echo "No checkpoint found, starting from random weights."
    input_arg=""
fi

echo "Normalizing training_data1 with training_postprocessor.py..."
python3 ./training_postprocessor.py --dir training_data0

TRAIN_DIR="${TRAIN_DIR:-training_data0}"

gdb --args ./build_release/rllm --train $input_arg \
    --train-dir "$TRAIN_DIR" \
    -o models/after_training.st \
     --filter guaranteed \
     --method random_line_random_len \
     --epochs 50 \
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
./build_debug.sh

echo "Formatting training_data/*.cpp with maximum line length..."
if compgen -G "training_data/*.cpp" > /dev/null; then
    clang-format -i --style='{BasedOnStyle: LLVM, ColumnLimit: 0}' training_data/*.cpp
else
    echo "No .cpp files found in training_data"
fi

# Resume from the latest checkpoint if one exists, otherwise start fresh.
latest_checkpoint=$(ls -t models/checkpoint-*.json 2>/dev/null | head -1)
if [ -n "$latest_checkpoint" ]; then
    echo "Resuming from $latest_checkpoint"
    input_arg="-i $latest_checkpoint"
else
    echo "No checkpoint found, starting from random weights."
    input_arg=""
fi

./build_debug/rllm --train $input_arg \
    -o models/after_training.json \
     --filter simple \
     --filter preprocessor \
     --filter iuring \
     --method window:32 --epochs 20

# ./build/rllm --train -i models/start.json \
#     -o models/after_training.json \
#      --filter simple --method three_tok --epochs 50
#

#./build/rllm --train -i models/start.json \
#    -o models/after_training.json \
#     --filter simple --method increasingly_longer --epochs 50
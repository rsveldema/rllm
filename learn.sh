
./build/rllm --train -i models/start.json \
    -o models/after_training.json \
     --filter iuring --method window:3 --epochs 5

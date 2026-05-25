
#./build/rllm --train -i models/start.json \
#    -o models/after_training.json \
#     --filter iuring --method window:3 --epochs 5

./build/rllm --train -i models/start.json \
    -o models/after_training.json \
     --filter simple --method three_tok --epochs 50

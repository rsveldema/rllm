
./build/rllm --train -i models/start.json \
    -o models/after_training.json \
    --filter iuring --filter simple --method window:6 --epochs 20

# ./build/rllm --train -i models/start.json \
#     -o models/after_training.json \
#      --filter simple --method three_tok --epochs 50
#

#./build/rllm --train -i models/start.json \
#    -o models/after_training.json \
#     --filter simple --method increasingly_longer --epochs 50
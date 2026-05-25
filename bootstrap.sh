
rm -f model.json

mkdir -p models

./build/rllm --train  --filter simple --method window:3 --epochs 5 -o models/start.json


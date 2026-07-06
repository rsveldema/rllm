# Training Utilities

`train_release.sh` and `train_debug.sh` try to resume from `RESUME_MODEL`, `models/after_training.st`, or the newest `models/checkpoint-*.st`.

Before auto-resuming, the scripts compare the checkpoint tokenizer vocabulary size with the generated runtime tokenizer. Incompatible automatic checkpoints are skipped and training starts from random weights. An explicitly supplied `RESUME_MODEL` must be compatible; if it is not, the script exits instead of silently ignoring the requested model.

This matters after tokenizer changes, such as adding the `INVALID` token, because old checkpoints have weight matrices with the previous vocabulary size.

## Learn Depth

`--learn-depth <N>` sets the number of gradient-update passes allowed for each training example before moving on. The default is `16`.

Increasing the value makes each example train longer before the next example is visited. Lower values move through the corpus more quickly.

For multi-token prediction, each example trains only the heads that have real future tokens. Short prefixes no longer train missing future heads toward `INVALID`.

## Learning Rate

`--learning-rate <R>` sets the base learning rate used during training. The default is `0.003`.

The effective per-update rate is divided by the number of transformer blocks, matching the previous hardcoded behavior.

Large values can saturate the clipped output weights and produce losses around `38400`, which means the target logit is clamped far below another token. For the current optimizer, prefer `0.003` to `0.01` as the base learning rate.

## Extending Checkpoints

When loading a checkpoint with fewer transformer blocks than the configured model, the saved blocks are loaded first and the extra blocks are initialized randomly. For example, `--layers 3 -i model-with-2-blocks.st` keeps the two saved blocks and appends one new block.

If the checkpoint has more blocks than `--layers`, the checkpoint block count is kept so existing models continue to resume without truncation.

## NaN Finding

NaN/range validation is disabled by default. Pass `--nan-finding` to enable the expensive runtime checks while debugging numerical instability.

# Training Utilities

`train_release.sh` and `train_debug.sh` try to resume from `RESUME_MODEL`, `models/after_training.st`, or the newest `models/checkpoint-*.st`.

Before auto-resuming, the scripts compare the checkpoint tokenizer vocabulary size with the generated runtime tokenizer. Incompatible automatic checkpoints are skipped and training starts from random weights. An explicitly supplied `RESUME_MODEL` must be compatible; if it is not, the script exits instead of silently ignoring the requested model.

This matters after tokenizer changes, such as adding the `INVALID` token, because old checkpoints have weight matrices with the previous vocabulary size.

The release training script includes `training_data1/preprocessor.cpp` via the `preprocessor` filter. This keeps prompt completions for prefixes such as `#`, `#in`, and `#if` trained on the dedicated preprocessor examples instead of only the incidental directives in larger C++ files.

## Learn Depth

`--learn-depth <N>` sets the number of gradient-update passes allowed for each training example before moving on. The default is `16`.

Increasing the value makes each example train longer before the next example is visited. Lower values move through the corpus more quickly.

For multi-token prediction, each example trains only the heads that have real future tokens. Short prefixes no longer train missing future heads toward `INVALID`.

Training diagnostics render unknown, missing, or out-of-range token IDs as `<UNK>` instead of aborting while formatting a log line. `Corpus::get_line` returns `std::nullopt` for those sequences.

## Learning Rate

`--learning-rate <R>` sets the base learning rate used during training. The default is `0.003`.

The effective per-update rate is divided by the number of transformer blocks, matching the previous hardcoded behavior.

Transformer dense projection updates additionally scale their weight updates by fan-in: transformer embedding projections scale by the embedding dimension, and the feed-forward down projection scales by the feed-forward dimension. Without that normalization, one example can shift downstream activations by hundreds or thousands of times the base learning rate because every input dimension contributes to the same output value.

Large values can still saturate the clipped output weights and produce losses around `38400`, which means the target logit is clamped far below another token. For the current optimizer, prefer `0.003` to `0.01` as the base learning rate.

If a previous run saturated or learned bad prompt completions, start a new release run with `FRESH_START=1 ./train_release.sh` so the script does not resume from the bad `models/after_training.st`.

## Extending Checkpoints

When loading a checkpoint with fewer transformer blocks than the configured model, the saved blocks are loaded first and the extra blocks are initialized randomly. For example, `--layers 3 -i model-with-2-blocks.st` keeps the two saved blocks and appends one new block.

If the checkpoint has more blocks than `--layers`, the checkpoint block count is kept so existing models continue to resume without truncation.

## NaN Finding

NaN/range validation is disabled by default. Pass `--nan-finding` to enable the expensive runtime checks while debugging numerical instability.

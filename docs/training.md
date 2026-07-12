# Training Utilities

`train_release.sh` and `train_debug.sh` try to resume from `RESUME_MODEL`, `models/after_training.st`, or the newest `models/checkpoint-*.st`.

Before auto-resuming, the scripts compare the checkpoint tokenizer vocabulary size with the generated runtime tokenizer. Incompatible automatic checkpoints are skipped and training starts from random weights. An explicitly supplied `RESUME_MODEL` must be compatible; if it is not, the script exits instead of silently ignoring the requested model.

This matters after tokenizer changes, such as adding the `INVALID` token, because old checkpoints have weight matrices with the previous vocabulary size.

The release training script includes `training_data1/preprocessor.cpp` via the `preprocessor` filter. This keeps prompt completions for prefixes such as `#`, `#in`, and `#if` trained on the dedicated preprocessor examples instead of only the incidental directives in larger C++ files.

## Learn Depth

`--learn-depth <N>` sets the number of gradient-update passes allowed for each training example before moving on. The default is `16`.

Increasing the value makes each example train longer before the next example is visited. Lower values move through the corpus more quickly.

After each epoch with validation enabled, training reports validation loss,
perplexity, and the average probability assigned to the correct token.
Perplexity is `exp(average loss)` and can be read as the effective number of
equally likely next-token choices, so lower is better. Correct-token probability
is the arithmetic mean of each evaluated target's softmax probability, so
higher is better.

For multi-token prediction, each example trains only the heads that have real future tokens. Short prefixes no longer train missing future heads toward `INVALID`.

Training diagnostics render unknown, missing, or out-of-range token IDs as `<UNK>` instead of aborting while formatting a log line. `Corpus::get_line` returns `std::nullopt` for those sequences.

## Epoch Size

`--epoch-size <N>` limits line-based training methods to `N` shuffled training lines per epoch. The default is all training lines. Values larger than the training split are clamped to the full split.

This is useful for faster validation/checkpoint feedback on large corpora. Window training keeps its existing window-based epoch behavior.

## Learning Rate

`--learning-rate <R>` sets the base learning rate used during training. The binary and training-script default for AdamW is `0.0003`.

The effective per-update rate is divided by the number of transformer blocks, matching the previous hardcoded behavior.

All learned parameters use AdamW with `beta1=0.9`, `beta2=0.999`,
`epsilon=1e-8`, and decoupled weight decay `0.01`. Gradients retain the existing
clipping. The old SGD-specific fan-in learning-rate scaling is not applied,
because AdamW normalizes updates by their second moment. Loading a checkpoint
restores its weights and starts fresh Adam moment buffers and optimizer step;
optimizer state is not serialized.

Large learning rates can still saturate the clipped weights and produce losses
around `38400`, which means the target logit is clamped far below another
token. Values inherited from the previous SGD configuration, such as `0.03`,
are too large for AdamW; start with `0.0003` and use validation trends when
tuning it.

If a previous run saturated or learned bad prompt completions, start a new release run with `FRESH_START=1 ./train_release.sh` so the script does not resume from the bad `models/after_training.st`.

## Extending Checkpoints

When loading a checkpoint with fewer transformer blocks than the configured model, the saved blocks are loaded first and the extra blocks are initialized randomly. For example, `--layers 3 -i model-with-2-blocks.st` keeps the two saved blocks and appends one new block.

If the checkpoint has more blocks than `--layers`, the checkpoint block count is kept so existing models continue to resume without truncation.

## NaN Finding

NaN/range validation is disabled by default. Pass `--nan-finding` to enable the expensive runtime checks while debugging numerical instability.

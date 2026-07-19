# Training Utilities

`train_release.sh` and `train_debug.sh` try to resume from `RESUME_MODEL`, `models/after_training.st`, or the newest `models/checkpoint-*.st`.

Every saved model has a sibling `<model filename>.training.json` containing the
training configuration. Resume both weights and settings with, for example,
`rllm --train --training-parameters models/checkpoint-123.st.training.json -i
models/checkpoint-123.st`. Options appearing after `--training-parameters`
override the archived value, which makes deliberate changes such as extending
the epoch count explicit while retaining all other settings.

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
Full training strings in `train.log` render newline and tab characters as `\\n`
and `\\t`, keeping each diagnostic on one physical log line.

Each batch progress message is also recorded as an object in an in-memory JSON
array. Entries include the method, epoch and item range, completion percentage,
loss, optimizer rounds, iteration counts, and forward/backward/apply timings.
After each model or checkpoint save, the complete array is written to
`train.json`, providing a valid snapshot for later visualization without doing
file I/O for every training batch.

Line-based training retains the newline token appended to every corpus line, so
line endings are learned as prediction targets. Whitespace within a line is also
preserved; for example, leading tab tokens in Python source remain part of the
training sequence.

Fresh models initialize attention, FFN, and LM-head matrices with Xavier-uniform
bounds derived from each matrix's fan-in and fan-out. Token embeddings use a
dimension-scaled uniform distribution with variance `1 / embedding_dimension`.
The defaults are `--weight-initializer xavier-input-projections`,
`--ffn-initializer xavier-input-projections`, and `--embedding-initializer
legacy-uniform`. Fresh-model runs log all three selected initializers in
`train.log`; loading a model does not reinitialize its parameters.

The `xavier-input-projections` weight/FFN profile applies Xavier initialization
only to Q/K/V and FFN gate/up matrices. Attention output, FFN down, and the LM
head use legacy scaling. `train_release.sh` selects this mixed profile and uses
legacy token embeddings.

## Epoch Size

`--epoch-size <N>` limits line-based training methods to `N` shuffled training lines per epoch. The default is all training lines. Values larger than the training split are clamped to the full split.

`--micro-batch-size <N>` must be between `1` and the compiled
`BatchIndex::MAX`. Larger values are rejected during argument parsing because
the packed batch metadata and output workspaces are statically bounded by that
index. The error message reports the active build's limit.

This is useful for faster validation/checkpoint feedback on large corpora. Window training keeps its existing window-based epoch behavior.

## Learning Rate

`--learning-rate <R>` sets the base learning rate used during training. The
binary default for AdamW is `0.0003`; `train_release.sh` currently selects
`0.00001` explicitly.

`--learning-rate-schedule constant|lowering|simulated_annealing` selects the learning-rate
implementation. `constant` keeps the configured rate unchanged for every
optimizer update. `lowering` (the binary default) applies the existing 5%
linear warmup and cosine decay.
`simulated_annealing` starts at
`--simulated-annealing-initial-multiplier <M>` times the configured rate
(default `50`) and remains constant within each epoch. Every
`--simulated-annealing-decay-epochs <N>` epochs (default `2`), it multiplies the
rate by `--simulated-annealing-decay-factor <F>` (default `0.8`) until
reaching `--simulated-annealing-min-multiplier <M>` times the configured base
rate (default `0.02`, or one fiftieth). The factor must be
greater than zero and less than one. `train_release.sh` currently selects
`simulated_annealing`.

The configured rate is the actual AdamW base rate and is not divided by the
number of transformer blocks. Each parameter tensor has independent AdamW
moments, so changing model depth does not silently change the CLI rate.

The resulting effective rate is adjusted by model depth using a linear profile.
`--layer-learning-rate-multiplier <M>` controls the output-head multiplier and
defaults to `1.05`. Token embeddings use `2-M`, and transformer blocks increase
gradually between those endpoints. Accepted values are `[1, 2)`. The profile is
symmetric with a mean multiplier of `1.0x`, so it changes the distribution of
learning across depth without increasing the model's average configured rate.

Window training applies a 5% linear learning-rate warmup followed by cosine
decay to 10% of the configured base rate. The schedule is calculated from the
number of windows, epochs, and allowed updates per window. For example, a base
rate of `0.00005` decays to `0.000005`. Loading a checkpoint starts a new
schedule because optimizer state and optimizer progress are not serialized.
The log reports the base and effective scheduled rate at startup, at warmup and
decay boundaries, and whenever the effective rate moves by at least 1% of its
peak value. This exposes schedule progress without logging every optimizer step.

`--window-stride <N>` controls the distance between primary next-token targets
within each corpus line and defaults to `1`. Each example records an explicit
context length followed by its primary target and up to three additional MTP
targets, without exceeding the configured maximum window size. A stride of `4`
matches the four MTP target heads and substantially reduces overlapping work;
a stride of `1` trains head zero at every token position. For example, it
includes an example whose context is `#in` and whose primary target is `clu`.

Window training uses the same deterministic 80/20 line split as line-based
training. Training and validation use the same boundary-preserving window
construction, so windows never join unrelated lines or bridge a held-out gap.
Before the first optimizer update, training records a validation-window baseline.
Every five minutes it reports held-out window loss, perplexity, average
correct-token probability, and validation duration together with the current
epoch and window progress.
It also performs and reports a held-out validation at the end of every epoch,
including epochs that finish before the five-minute interval.
Window validation reports head-zero next-token loss as its primary metric, plus
the aggregate loss and individual loss for all four MTP heads. Checkpointing and
early stopping use head-zero loss so they optimize the completion objective and
are directly comparable with reported training loss. The baseline and each
end-of-epoch validation also report the five worst individual predictions. Each
diagnostic includes loss, expected and predicted tokens with their probabilities,
the MTP head, and the decoded input context.
Each batch progress line also reports `training loss`, the mean primary
next-token cross-entropy for the examples and optimizer rounds in that batch.
An epoch-level training-window mean is reported separately. These online
training values are expected to be noisier than loss over the full held-out
validation-window split.

Batched backpropagation optimizes the mean loss over all active `(example, MTP
head)` predictions. The softmax deltas are divided by that prediction count
before output, transformer, and embedding gradients are accumulated, clipped,
and passed to AdamW. This keeps update scale independent of micro-batch size,
short-window head count, and examples that converge before the final round.

`--method reverse_window` visits the boundary-preserving windows from the end
toward the start. It uses the current maximum window size
(default `2`); `--method reverse_window:<N>` selects the size inline. Unlike
regular window training, reverse-window traversal is not shuffled. Window
stride continues to apply within each line.

Window training saves `models/checkpoint-best-window.st` whenever end-of-epoch
head-zero validation loss improves by at least `1e-4`. It stops after three consecutive
epochs without improvement and restores that best checkpoint before the final
model is saved. Timed intra-epoch validation remains diagnostic and does not
advance early-stopping patience.
Every tenth completed batch logs `HH:MM:SS` estimates of the wall-clock time
remaining in the current epoch and until all planned epochs finish. The
estimates use elapsed epoch time and the completed example/window fraction, so
checkpointing and validation time already spent in that epoch are reflected in
later estimates. Early stopping may finish before the all-epochs estimate.

At startup, `train.log` reports estimated transformer GPU memory in MB per layer and
multiplied by the configured layer count. The estimate includes block weights,
Adam optimizer state, per-layer forward workspaces and gradient accumulators,
and the correctly apportioned shared backward workspace. It excludes non-block
buffers and Vulkan allocator overhead.

The startup log also reports GPU memory required per compiled batch slot and in
total across `BatchIndex::MAX`. This batch-slot estimate covers the batched
output workspace and per-example packed-row boundaries. Token-level packed
metadata and transformer activations are instead bounded by `PositionIndex::MAX`
and are not multiplied by the number of batch slots.

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

# Early stopping

Training stops after three consecutive epochs without validation-loss improvement by default. Pass
`--disable-early-stopping` to run every epoch requested with `--epochs`. Validation, best-checkpoint
saving, and restoration of the best checkpoint remain enabled.
`--disable-example-convergence` keeps every example active instead of removing
examples that cross the per-example convergence threshold. This is intended for
optimizer-stability and memorization diagnostics.

The first optimizer update of each window-training epoch logs per-group
diagnostics: pre-clipping gradient maximum, clipping percentage, Adam update
RMS/maximum, and applied weight-update RMS/maximum. The groups are embeddings,
individual transformer layers, and individual output heads.

Before each AdamW update, gradients are globally L2-clipped to norm `1.0`
independently for each of those parameter groups. The existing element-wise
clamp remains as a secondary safety check. Diagnostics also report the original
group norm and the global scaling factor that was applied.

The first backward pass of each epoch also logs hidden-gradient RMS, maximum,
and L2 norm at the output heads and after every transformer block. Within each
block it additionally logs the FFN down-projection output, the SwiGLU gate and
upstream branches, both sides of the FFN RMSNorm backward pass, the attention
output-projection result, attention `d_V`, aggregate `d_scores` and `d_raw`
statistics across all heads, attention `d_Q` and `d_K`, and both sides of the
attention RMSNorm backward pass. It also reports the stored forward Q/K/V
activations and per-head `d_scores`, making an anomalous head or forward value
visible when score gradients spike. These additional forward and per-head
diagnostics are compiled when the CMake option `DEBUG_ATTENTION_DIAGNOSTICS` is
enabled (the default); disable it to remove their runtime and binary overhead.
The layer number and stage name on each line identify where amplification first
appears without adding per-batch log volume.

# Training Utilities

Launch training by passing a JSON configuration file:

```bash
./train.py config-6.json
./train.py config-6.json --fresh-start
./train.py config-6.json --resume-model models-6/checkpoint-best-window.st
```

The configuration owns the build type, model directory, layer
count, window size and stride, learn depth, source mixture, comment processing,
CMake arguments, and rLLM training arguments. `config-6.json` describes the
six-layer release model. The launcher does not read training options from
environment variables. Resume behavior is selected with the mutually exclusive
`--fresh-start` and `--resume-model PATH` command-line options; with neither,
the launcher selects a compatible checkpoint automatically.

`--train-dir <path>` may be repeated. Add `:<weight>` to define the desired
share of epoch draws for each source, for example `--train-dir curriculum:0.2
--train-dir training_data2:0.8`. Weights are normalized across non-empty
sources, so small curriculum corpora can remain represented without being
overwhelmed by a larger source. Sampling cycles through shuffled source windows
before repeating them. Unweighted directories default to weight `1`.

The Python training launcher uses `training_data0`, the `curriculum/grammar`,
`curriculum/syntax`, `curriculum/comments`, and `curriculum/systems` categories,
and `training_data2` as a weighted mixture. Change `sources` in the JSON to use
a different corpus or weighting.

`./train.py config-6.json` stores artifacts in the configured model directory.
Unless `--fresh-start` or `--resume-model` is supplied, it tries to resume from
the matching model directory or—when starting a depth-increasing upgrade—the
best checkpoint or final model in the deepest available `models-N/` below the
target depth. Window size, stride, and learn depth are explicit JSON fields; the
six-layer configuration uses 96, 48, and 3 respectively.

Every saved model has a sibling `<model filename>.training.json` containing the
training configuration. Resume both weights and settings with, for example,
`rllm --train --training-parameters models/checkpoint-123.st.training.json -i
models/checkpoint-123.st`. Options appearing after `--training-parameters`
override the archived value, which makes deliberate changes such as extending
the epoch count explicit while retaining all other settings.

Pass `--upgrade-mode --layers N` while loading a shallower checkpoint to append
newly initialized transformer layers up to depth `N`. By default, the complete
expanded model remains trainable. Add `--freeze-old-blocks` to keep only the
original transformer layers read-only. The global token embeddings, output
heads, and appended layers remain trainable, and gradients pass through the
read-only layers to reach the embeddings. Upgrade mode rejects targets that are
shallower than their source checkpoint.
Extending a checkpoint starts a fresh optimization phase: it resets the saved
epoch and window cursor, Adam moments and bias-correction step, and learning-rate
schedule. The resized model therefore starts at epoch 0 with the configured base
learning rate. Loading a checkpoint that is already at the requested depth
remains a normal resume and preserves its training state.

Checkpoints persist the frozen boundary. Loading one normally restores its
read-only blocks; pass `--all-blocks-read-write` to clear that boundary and let
the complete model participate in learning again. The next checkpoint persists
the cleared boundary. At startup, `train.log` and the console report the exact
number of read-only and read-write transformer blocks.

When the Python launcher finds a checkpoint to resume, it supplies
`--upgrade-mode --freeze-old-blocks` by default, using the layer count configured
in JSON. A fresh-start run uses a random model at that configured depth without
upgrade mode. Add `--all-blocks-read-write` to the JSON `training_arguments` to
override the default freeze for a resumed model.

Safetensors checkpoints created during training also embed resumable optimizer
state: Adam first and second moments for embeddings, transformer parameters,
and output heads; the optimizer step; the configured base and depth-scaled
learning-rate settings; the schedule type and planned step count; and the
current lowering or simulated-annealing position. Loading such a checkpoint
restores this embedded state after parsing the command-line configuration, so
the next optimizer update uses the same bias correction and effective learning
rate. Window-training checkpoints additionally store the epoch/window cursor
and the shuffle generator state, allowing a timed mid-epoch checkpoint to
continue at the next unprocessed micro-batch with the same window order. Older
weight-only checkpoints remain supported and start with fresh optimizer and
schedule state.

Line-based checkpoints resume at epoch granularity. An end-of-epoch checkpoint
continues with the following epoch; a timed checkpoint written during an epoch
restarts that epoch with the saved pre-shuffle generator state.

The saved window count distinguishes mid-epoch checkpoints from completed
epochs. If the corpus changes, an incomplete epoch restarts at cursor zero;
epoch-complete checkpoints continue at the following epoch. Checkpoints written
before the window-count metadata was introduced recover a non-batch-aligned
cursor as the old end-of-epoch representation.

Before auto-resuming, the launcher compares the checkpoint tokenizer vocabulary size with the generated runtime tokenizer. Incompatible automatic checkpoints are skipped and training starts from random weights. A model supplied through `--resume-model` must be compatible; if it is not, the launcher exits instead of silently ignoring the requested model.

This matters after tokenizer changes, such as adding the `INVALID` token, because old checkpoints have weight matrices with the previous vocabulary size.

The supplied configuration uses `training_data2` and does not set a filename
filter, so every program in that directory is included.
Edit the configuration's `sources` array to select another corpus. Passing one or more
`--filter` options explicitly narrows loading to filenames containing any of
those values. When using `curriculum`, its `preprocessor.cpp` supplies
dedicated completion examples for prefixes such as `#`, `#in`, and `#if`.
Corpus loading records processed files and tokenization errors in
`tokenization.log` beside the output model. The standard scripts therefore
write it to `models/tokenization.log`. Per-token match tracing is disabled by
default so loading a large unfiltered corpus does not perform a synchronous log
write for every token candidate.

Unit tests set `RLLM_MODEL_DIR=test_models` and clean only that directory.
Timed checkpoints, best checkpoints, training progress, and tokenization logs
created by tests never use the production `models/` directory.
The Python training launcher requests a timed checkpoint every
600 seconds, for at most six timed checkpoints per hour of uninterrupted
training.

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
Training creates `train.json` beside the output model immediately as a valid
empty JSON array, before potentially lengthy baseline validation. After each
model or checkpoint save, the complete progress array is written there.
Training batches also flush it at most every 15 seconds. Long validation passes
write `validation_progress` entries and text-log updates every 15 seconds with
the completed count, running head-zero loss, elapsed time, and ETA. For example,
The Python launcher output `models-<layers>/after_training.st` uses
`models-<layers>/train.json` and `models-<layers>/train.log`.
Restarting training loads and extends the existing JSON array, while truncating
the text log for the new process. This preserves one continuous structured
history while keeping `train.log` scoped to the current run and retaining a
valid JSON snapshot without doing file I/O for every training batch.
Completed timed checkpoints write a `checkpoint` entry with phase `timed`, the
epoch and window cursor, total windows trained, and duration in milliseconds and
seconds. The same duration is reported in `train.log` and on the console.
Validation records include head-zero loss, perplexity, all-MTP loss, and the
average correct-token probability. Generate `training_metrics.png` with an
overlay of every `models-*/train.json` with
`python visualize_training.py`, or choose paths with
`python visualize_training.py path/to/train.json -o plot.png`.
The generated figure also plots the normalized global gradient norm for every
recorded optimizer parameter group. The plot opens interactively by default;
pass `--no-show` for headless or save-only use.

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
head use legacy scaling. `train.py` selects this mixed profile and uses
legacy token embeddings.

## Epoch Size

`--epoch-size <N>` limits line-based training methods to `N` shuffled training lines per epoch. The default is all training lines. Values larger than the training split are clamped to the full split.

`--max-validation-windows <N>` deterministically samples at most `N` evenly
spaced held-out windows. Its default is 4,096; the shared release/debug training
wrapper sets that value explicitly.

`--validation-worst-count <N>` controls how many of the highest-loss validation
predictions are written to `train.log`. Its default is `5`; the shared
Python training launcher sets it to `100`.

`--micro-batch-size <N>` must be between `1` and the compiled
`BatchIndex::MAX`. Larger values are rejected during argument parsing because
the packed batch metadata and output workspaces are statically bounded by that
index. The error message reports the active build's limit.

This is useful for faster validation/checkpoint feedback on large corpora. Window training keeps its existing window-based epoch behavior.

At startup, `train.log` reports the number of source tokens assigned to the
training side of the deterministic split. Window training also reports token
occurrences across all generated windows for one full epoch; that count includes
overlap between windows. Validation tokens are excluded from both counts. The
log records the configured maximum window size in tokens and the within-file
stride before constructing the training and validation windows.

## Learning Rate

`--learning-rate <R>` sets the base learning rate used during training. The
binary default for AdamW is `0.0003`; `train.py` currently selects
`0.00001` explicitly.

`--learning-rate-schedule constant|lowering|simulated_annealing` selects the learning-rate
implementation. `constant` keeps the configured rate unchanged for every
optimizer update. `lowering` (the binary default) applies a configurable linear
warmup and cosine decay. `--warmup-percent <P>` specifies warmup as a percentage
of the planned epochs and accepts values in `(0, 100]`; its default is `5`.
`simulated_annealing` starts at
`--simulated-annealing-initial-multiplier <M>` times the configured rate
(default `50`) and remains constant within each epoch. Every
`--simulated-annealing-decay-epochs <N>` epochs (default `2`), it multiplies the
rate by `--simulated-annealing-decay-factor <F>` (default `0.8`) until
reaching `--simulated-annealing-min-multiplier <M>` times the configured base
rate (default `0.02`, or one fiftieth). The factor must be
greater than zero and less than one. `train.py` currently selects
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

Window training applies the configured linear learning-rate warmup followed by
cosine decay to 10% of the configured base rate. The schedule is calculated
from the number of windows, epochs, and allowed updates per window, so a 1%
warmup spans 1% of the planned epochs. The Python launcher selects
`--warmup-percent 1`. For example, a base rate of `0.00005` decays to
`0.000005`. Current checkpoints restore the optimizer, warmup percentage, and
schedule position; legacy checkpoints use the CLI/default warmup percentage.
The log reports the base and effective scheduled rate at startup, at warmup and
decay boundaries, and whenever the effective rate moves by at least 1% of its
peak value. This exposes schedule progress without logging every optimizer step.

`--window-stride <N>` controls the distance between supervised hidden rows and
defaults to `1`. A window contains up to the configured token count and performs
one causal transformer pass. Head zero predicts the following token from every
stride-selected row; higher MTP heads predict the additional available future
tokens. Adjacent windows overlap by one token, so stride `1` covers every
next-token boundary exactly once without rebuilding every prefix.

Window training deterministically reserves 20% of the windows within each file
for validation. Every file large enough to produce at least two windows
therefore contributes to both training and validation; shorter files remain
training-only. A window may span multiple lines within its source file,
including newline tokens, but never crosses into another file.
Every training and validation window begins with one of six language tokens:
`<LANG_CPP>`, `<LANG_C>`, `<LANG_PYTHON>`, `<LANG_RUST>`, `<LANG_JAVA>`, or
`<LANG_SHELL>`. The marker is input context rather than a prediction target and
occupies one position inside the configured window size. Files with an unknown
extension currently use the C++ marker.

The lexer replaces source comment delimiters with
`<LINE_COMMENT_START> ... <LINE_COMMENT_END>` or
`<BLOCK_COMMENT_START> ... <BLOCK_COMMENT_END>` around C, C++, Java, Rust,
Python, and shell comment contents. For example, Python `# hello` becomes
`<LINE_COMMENT_START> hello <LINE_COMMENT_END>` in
the model token stream. It recognizes `//`, `#`, and `/* ... */`
outside quoted strings, carries block-comment state across lines, and supports
nested Rust block comments. A window beginning in the middle of a block comment
receives `<BLOCK_COMMENT_START>` after its language prefix. Metadata prefixes remain
inside the configured window size, so the minimum supported window is four
tokens.
`<LINE_COMMENT_END>` replaces the source newline rather than preceding a
separate newline token; prompt rendering converts it back to `\n`.

Prompt mode uses the C++ marker by default. Use `/language <name>` (or `/lang
<name>`) to select `cpp`, `c`, `python`, `rust`, `java`, or `shell`; subsequent
prompts and hidden-state probes receive that language marker.
Corpus lines and generated windows use compact ordinary CPU storage. Vulkan
host-visible staging buffers are allocated only for reusable GPU upload
objects, avoiding one driver allocation per window when large corpora are
loaded.
Generated Vulkan dispatches allocate descriptor sets from a reusable
per-queue arena. The arena is reset only after that queue becomes idle, so
forward and backward passes do not create and destroy a descriptor pool for
every kernel invocation.
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
Held-out windows reserve up to four trailing tokens beyond their validation
context, so each available MTP head is evaluated against a real future token.
Short final windows contribute only the heads for which a future token exists.
When the per-file reservation produces more windows than `--max-validation-windows`,
validation distributes that budget across the source files, including at
least one window from every file when the budget permits, and samples evenly
within each file. This keeps baseline, periodic, and end-of-epoch validation
cost bounded while covering files of different sizes; training still uses
every training window.
Validation packs those windows into the configured micro-batch size and uses
the same batched transformer and output-loss kernels as training. Worst-token
diagnostics rerun only the five highest-loss predictions individually.
Each batch progress line also reports `training loss`, the mean primary
next-token cross-entropy for the examples and optimizer rounds in that batch.
An epoch-level training-window mean is reported separately. These online
training values are expected to be noisier than loss over the full held-out
validation-window split.

Batched backpropagation optimizes the mean loss over all active `(window row,
MTP head)` predictions. The softmax deltas are divided by that prediction count
before output, transformer, and embedding gradients are accumulated, clipped,
and passed to AdamW. This keeps update scale independent of micro-batch size,
short-window head count, and examples that converge before the final round.

`--method reverse_window` visits the boundary-preserving windows from the end
toward the start. It uses the current maximum window size
(default `2`); `--method reverse_window:<N>` selects the size inline. Unlike
regular window training, reverse-window traversal is not shuffled. Window
stride continues to apply within each line.

Window training saves `checkpoint-best-window.st` in the configured model
artifact directory whenever end-of-epoch
head-zero validation loss improves by at least `1e-4`. It stops after three consecutive
epochs without improvement and restores that best checkpoint before the final
model is saved. Timed intra-epoch validation remains diagnostic and does not
advance early-stopping patience.
`--validation-interval <seconds>` controls timed intra-epoch validation and
defaults to 1,800 seconds. The Python launcher sets it explicitly
to 1,800 seconds. Its timer restarts after evaluation completes; baseline and
end-of-epoch validation are unchanged.
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

If a previous run saturated or learned bad prompt completions, pass
`--fresh-start` so the launcher does not resume from the configured
`after_training.st`.

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

The corresponding training-progress entry in `train.json` contains an
`optimizer_diagnostics` array with each `parameter_group`, its
`global_gradient_norm`, `global_clip_scale`, and the resulting
`normalized_global_gradient_norm`. Because diagnostics are sampled on the first
optimizer update of an epoch, the array appears only on the progress entry that
contains that update.

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
`train.py` explicitly enables this option and leaves per-epoch optimizer
and backward-pass diagnostic collection enabled so `train.json` contains the
optimizer diagnostic samples. Additional CMake options can be passed directly
to `build_release.sh`.

The `build_type` field selects whether the launcher builds and runs
`build_debug/rllm` or `build_release/rllm`. All other runtime arguments come
from the same JSON file. For example, change the value following `--epochs` in
`training_arguments` to alter the epoch count.

The Python launcher removes comments by default. The default source mix omits
the dedicated `curriculum/comments` corpus, and preprocessing creates temporary
copies of the remaining corpora with C/C++ and Python comments removed. Text
that merely resembles a comment inside a string literal is preserved. The
source training directories remain unchanged and the temporary copies are
removed when the launcher returns. Set `strip_comments` to `false` to preserve
comments in the selected source corpora.

Before the corpus is loaded, files in the selected training directory are
normalized by `training_postprocessor.py`. Python files have every
complete group of four leading spaces converted to a literal tab. The runtime
tokenizer preserves each resulting tab as `TokenID::TOK_TAB`, allowing Python
block indentation to participate in training instead of being discarded with
ordinary spaces.

For controlled resume experiments, `--reset-optimizer-state` keeps the loaded
weights, training cursor, and learning-rate position but clears all Adam moments
and restarts the bias-correction step. `--restart-learning-rate-schedule` keeps
the loaded weights, cursor, and Adam state, discards the checkpoint's schedule
position and total-step count, and recomputes a fresh schedule from the current
training configuration.
The options are independent and can be combined.
The layer number and stage name on each line identify where amplification first
appears without adding per-batch log volume.

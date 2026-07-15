# Microbatching

`--micro-batch-size 1` trains one example per packed batch.

Values greater than one group examples into batches for every training method,
including sliding windows, fixed-length prefixes, increasingly longer
prefixes, random prefixes, and full lines. Each training
step accumulates gradients for every valid example in the batch, then applies
one summed update. This intentionally increases the effective update magnitude
with the number of active examples, which preserves the faster convergence seen
with micro-batching. For random-prefix training, each prefix is selected once
and reused for all training steps in that batch. Window training
uses the configured fixed window length at every sliding start position before
the packed batch is built. Start positions honor `--window-stride`, are shuffled
each epoch, and use an epoch-varying initial offset.

True tensor batching uses a packed ragged row axis. Each example owns a
contiguous row interval, positional indices restart at zero for every example,
and causal attention is restricted to keys in the query's interval. This is a
block-diagonal causal mask; examples in a micro-batch must never attend to one
another.

Batched output scoring remains GPU-resident. Softmax, cross-entropy, and delta
generation operate directly on the batched logits; training reads back only the
small primary-head loss vector used for convergence decisions. The Vulkan
reduction kernels use tkernel workgroup reductions to limit global atomics.

The progress log reports `iterations total` as the sum of the iterations used
by all examples, `avg .../line` as that total divided by the batch size, and
`rounds` as the number of batch-wide optimizer rounds. Consequently, the total
may exceed the per-example maximum even though no individual example did.

# Microbatching

`--micro-batch-size 1` trains one example per packed batch.

Values greater than one group all-position windows into batches. Each training
step accumulates gradients for every valid example in the batch, then applies
one summed update. This intentionally increases the effective update magnitude
with the number of active examples, which preserves the faster convergence seen
with micro-batching. One causal transformer pass per window supplies loss at
every stride-selected row whose target is present. Windows may span newlines,
but are built independently for each source file. Adjacent windows overlap by
one boundary token, so stride 1 supervises every next-token boundary once
without recomputing all prefixes. Windows are shuffled each epoch.

True tensor batching uses a packed ragged row axis. Each example owns a
contiguous row interval, positional indices restart at zero for every example,
and causal attention is restricted to keys in the query's interval. This is a
block-diagonal causal mask; examples in a micro-batch must never attend to one
another.

Batched output scoring remains GPU-resident. Valid hidden rows are gathered in
chunks, and softmax, cross-entropy, and delta generation operate directly on
their batched logits. Training reads back only primary-head losses used for
reporting and convergence decisions. The Vulkan
reduction kernels use tkernel workgroup reductions to limit global atomics.

Input-embedding backpropagation is also GPU-resident. Packed hidden gradients
are accumulated into token rows with a tkernel-backed atomic kernel, and the
embedding Adam moments, global clipping, and weight update remain on the device.
Embedding tensors are copied to the host only for explicit serialization or
inspection.

The progress log reports `iterations total` as the sum of the iterations used
by all examples, `avg .../line` as that total divided by the batch size, and
`rounds` as the number of batch-wide optimizer rounds. Consequently, the total
may exceed the per-example maximum even though no individual example did.

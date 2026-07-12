# Microbatching

`--micro-batch-size 1` uses the existing single-example random-line training
path.

Values greater than one group random-line examples into batches. Each training
step accumulates gradients for every valid example in the batch, then applies
one summed update. This intentionally increases the effective update magnitude
with the number of active examples, which preserves the faster convergence seen
with micro-batching. Random prefixes are selected once and reused for all
training steps in that batch.

True tensor batching uses a packed ragged row axis. Each example owns a
contiguous row interval, positional indices restart at zero for every example,
and causal attention is restricted to keys in the query's interval. This is a
block-diagonal causal mask; examples in a micro-batch must never attend to one
another.

The progress log reports `iterations total` as the sum of the iterations used
by all examples, `avg .../line` as that total divided by the batch size, and
`rounds` as the number of batch-wide optimizer rounds. Consequently, the total
may exceed the per-example maximum even though no individual example did.

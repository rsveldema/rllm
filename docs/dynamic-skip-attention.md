# Dynamic Skip Attention

Dynamic Skip Attention (DSA) lets transformer execution `i` build its input
from residuals produced by earlier executions. Routing is conditioned on the
token at each sequence row.

The learned parameter is a shared `token × source-execution` logit table.
Before an execution runs, the eligible prefix of that token's row is normalized
with a softmax. The resulting weights sum to one, so adding older residual
paths does not increase the residual scale.

Logits are initialized with a step of 20 between consecutive source executions.
Consequently the immediate predecessor receives effectively all probability at
startup, while every earlier source remains differentiable and can be learned.
Gradients are routed both to every selected residual and to the token's routing
logits.

The table is stored as `dynamic_skip_attention.logits` in safetensors models
and as `dynamic_skip_attention.logits` in JSON models. Adam state is stored as
`training.dynamic_skip_attention.adam_first` and
`training.dynamic_skip_attention.adam_second`. DSA tensors are mandatory;
pre-DSA checkpoints are rejected rather than upgraded implicitly.

The routing table supports up to 32 transformer executions. In looped models,
each occurrence in the expanded execution order has its own routing index,
even when multiple occurrences share the same physical block weights. A
looped execution can therefore route to any earlier execution, including
residuals from a previous pass through the loop.

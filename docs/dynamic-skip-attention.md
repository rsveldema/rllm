# Dynamic Skip Attention

Dynamic Skip Attention (DSA) lets transformer layer `L` build its input from
the output residuals of layers `1` through `L-1`. Routing is conditioned on the
token at each sequence row.

The learned parameter is a shared `token × source-layer` logit table. Before a
layer runs, the eligible prefix of that token's row is normalized with a
softmax. The resulting weights sum to one, so adding older residual paths does
not increase the residual scale.

Logits are initialized with a step of 20 between consecutive source layers.
Consequently the immediate predecessor receives effectively all probability at
startup, while every earlier source remains differentiable and can be learned.
Gradients are routed both to every selected residual and to the token's routing
logits.

The table is stored as `dynamic_skip_attention.logits` in safetensors models
and as `dynamic_skip_attention.logits` in JSON models. Adam state is stored as
`training.dynamic_skip_attention.adam_first` and
`training.dynamic_skip_attention.adam_second`. DSA tensors are mandatory;
pre-DSA checkpoints are rejected rather than upgraded implicitly.

The compiled model supports up to 32 physical transformer layers. Existing
looped-layer models retain their loop transition behavior and currently bypass
DSA because repeated physical layers do not have an unambiguous source-layer
index in the shared table.

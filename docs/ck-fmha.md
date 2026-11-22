# CK-Tile fused attention

rllm carries AMD Composable Kernel as the `third_party/composable_kernel` Git
submodule. Initialize it with:

```sh
git submodule update --init --recursive
```

HIP builds enable `RLLM_ENABLE_CK_FMHA` by default and require an explicit GPU
target. It can be disabled with `-DRLLM_ENABLE_CK_FMHA=OFF`:

```sh
./build_release.sh -DGPU_TARGETS=gfx1201
```

The build intentionally generates only the kernel family needed by the model:
FP16 Q/K/V, 64 values per head, packed/group mode, causal masking, saved
log-sum-exp, and no attention bias or dropout. Backward generation includes the
matching group-mode stages. This avoids compiling CK's thousands of example
variants.

When enabled, the runtime dispatches causal FP16 attention forward passes to CK
for both single sequences and packed variable-length batches. CK writes FP16
attention outputs and FP32 log-sum-exp values; rllm converts the output to FP32
for the output projection. The existing FP32 backward path remains active and
recomputes attention probabilities because CK forward does not materialize
them. If CK cannot select a compatible forward kernel, rllm falls back to its
in-tree attention implementation.

A single dense attention-probability scratch buffer is preallocated with the
trainer workspace and reused by every transformer layer. Backward recomputes
the current layer's probabilities into that buffer, so no per-layer copy or
training-time GPU allocation is required.

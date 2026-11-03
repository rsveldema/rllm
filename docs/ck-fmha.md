# CK-Tile fused attention

rllm carries AMD Composable Kernel as the `third_party/composable_kernel` Git
submodule. Initialize it with:

```sh
git submodule update --init --recursive
```

Configure `RLLM_ENABLE_CK_FMHA=ON` together with HIP offload and an explicit
GPU target:

```sh
./build_release.sh -DRLLM_ENABLE_CK_FMHA=ON -DGPU_TARGETS=gfx1201
```

The build intentionally generates only the kernel family needed by the model:
FP16 Q/K/V, 64 values per head, packed/group mode, causal masking, saved
log-sum-exp, and no attention bias or dropout. Backward generation includes the
matching group-mode stages. This avoids compiling CK's thousands of example
variants.

The generated CK kernels are not dispatched by the runtime yet. The active
in-tree attention path stores Q, K, and V projection outputs and their RoPE
results in FP16. Attention dot products, softmax probabilities, output
accumulation, and gradients remain FP32. A runtime adapter is still needed
before the CK FMHA kernels can replace this path.

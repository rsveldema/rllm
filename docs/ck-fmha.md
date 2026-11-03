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

The existing FP32 attention implementation remains the runtime path until the
FP16 conversion buffers, packed sequence-start metadata, and forward/backward
dispatch adapter are connected. Enabling this option currently builds and links
the CK kernels; it does not change numerical behavior.

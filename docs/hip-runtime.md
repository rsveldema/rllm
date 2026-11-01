# HIP runtime

HIP offload uses one non-blocking `hipStream_t` per `HipQueue`. Device storage
is allocated with `hipMalloc`, CPU staging storage uses `hipHostMalloc`, and
copies/zeroing are enqueued on the selected stream. A host read or an explicit
`HipQueue::wait()` synchronizes that stream.

The kernel compiler continues to apply optimizations through `.tkernel` AST
rewrites. Backend lowering emits native `.hip` sources, compiles them to
`.hsaco` code objects with `hipcc --genco`, and generates C++ dispatch stubs
that launch `rllm_kernel` through the HIP module API.

Configure offload with `-DOFFLOAD_BACKEND=hip`. ROCm/HIP and `hipcc` must be
installed. `build_debug.sh` and `build_release.sh` select HIP offload and the
sequential CPU scheduling backend by default.

`rllm --hip-device <substring>` selects a preferred HIP device by name. The
`RLLM_HIP_DEVICE_INDEX` environment variable controls the device shown by the
provider diagnostics.

`hip_runtime::ScopedQueueOffset` temporarily offsets queue-relative lookups
from `hip_runtime::get_queue(index)`, preserving the existing micro-batch queue
rotation behavior.

## Conservative BLAS lowering

With optimizations enabled, the kernel compiler recognizes the exact FP32
contraction `C[i, j] += sum_k A[k, i] * B[k, j]` and dispatches it through
`hipblasSgemm`. The matcher operates on the kernel AST and requires canonical
two-dimensional indexing, read-only inputs, a mutable output, and the original
accumulation semantics (`beta = 1`).

Kernels containing an epilogue, clamp, cast, bias, additional output, mixed
element types, or non-canonical indexing are rejected by the matcher and use
the regular generated HIP kernel. Passing `--no-optimizations` also disables
BLAS recognition.

The FFN forward path retains its normalized activation cache in FP32 for
backward weight gradients and creates an FP16 projection mirror. Exact
FP16-by-FP16 `A * B^T` projections with FP32 accumulation and output are
lowered to `hipblasGemmEx`; gate and up projections use this path independently.

SwiGLU activations are computed and clamped to `[-10000, 10000]` in FP32, then
stored as FP16. The forward down projection and backward weight-gradient
kernels read this FP16 cache while retaining FP32 sums and gradient storage.

An exact FP16-by-FP16 `A * B^T` whose only epilogue is the standard
`[-10000, 10000]` clamp is split into two queued operations: an
FP32-accumulating `hipblasGemmEx` followed by a generated elementwise HIP clamp
kernel. The split is performed by `blas_clip_abt.tkernel`; other epilogues and
clamp bounds continue to use the original generated kernel.


# Vulkan Runtime

Generated Vulkan kernel stubs submit compute dispatches asynchronously. They do
not wait for queue idle after every dispatch. Work on the same queue remains
ordered by submission, and explicit host transfers or `VulkanQueue::wait()`
synchronize when CPU-visible results or resource cleanup are needed.

The `rllm` command accepts `--vulkan-device <substring>` to select the physical
device used for computation. Matching is case-insensitive, and startup fails if
no enumerated Vulkan device name contains the requested text.

This keeps short training kernels from forcing a CPU/GPU round trip after every
operation.

`VBaseDeviceBuffer::zero()` is also wait-free. Zero fills are recorded into a
per-queue batch and flushed before the next normal command buffer allocation,
before `VulkanQueue::wait()`, or when the batch reaches its fill limit. This
lets accumulator resets coalesce many fills into fewer queue submissions while
preserving same-queue ordering before dependent kernels.

`vulkan_runtime::ScopedQueueOffset` temporarily offsets queue-relative lookups
from `vulkan_runtime::get_queue(index)`. Micro-batched training uses this to
rotate consecutive optimizer micro-batches across the available Vulkan queues
while preserving existing queue-relative call sites.

Vulkan builds must not use the FastFork `PARFOR*` backend. Configure with
`PARALLEL_BACKEND=sequential` (the default in `build_debug.sh` and
`build_release.sh`) or `PARALLEL_BACKEND=openmp`. Combining FastFork with
`USE_VULKAN_OFFLOAD` produces a compile-time error because FastFork tasks can
migrate Vulkan dispatches away from their queue-affine caller.

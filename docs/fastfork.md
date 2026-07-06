# FastFork Runtime

FastFork initializes its worker count from the available hardware threads by default.

When rllm is built with the Vulkan offload backend, the default FastFork worker count is capped at 8 threads. This keeps CPU-side scheduling from overfeeding Vulkan command submission on systems where the hardware thread count is much higher than the number of useful Vulkan queues.

Explicit calls to `fastfork::set_num_threads()` still override the default in all builds.

# Vulkan Runtime

Generated Vulkan kernel stubs submit compute dispatches asynchronously. They do
not wait for queue idle after every dispatch. Work on the same queue remains
ordered by submission, and explicit host transfers or `VulkanQueue::wait()`
synchronize when CPU-visible results or resource cleanup are needed.

This keeps short training kernels from forcing a CPU/GPU round trip after every
operation.

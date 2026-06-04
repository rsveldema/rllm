#pragma once

#include <cstddef>

#if defined(USE_VULKAN_OFFLOAD) || defined(USE_HIP_OFFLOAD)
#define RLLM_DEVICE_POINTER_HAS_OFFLOAD 1
#else
#define RLLM_DEVICE_POINTER_HAS_OFFLOAD 0
#endif

enum class IMemorySpaceType {
    HOST,
    VULKAN,
    HIP
};


/** Allocate/deallocate memory in a specific memory space.
 *
 * When we have a GPU / accelerator backend, this abstracts over host vs device memory allocation.
 * For CPU-only builds this is a no-op wrapper around standard heap allocation.
 * For Vulkan offload builds this will allocate from the appropriate Vulkan memory heap and manage
 * staging buffers as needed. For HIP offload builds this will allocate from host or device memory
 * as appropriate. In both offload cases the implementation will also handle any necessary
 * synchronization to ensure memory visibility between host and device.
 *
 * This will at bootup allocate a large block of memory for the entire model and training state, and then sub-allocate from that for individual tensors. This is more efficient than doing many small device allocations, and also allows us to implement a simple free list to reuse freed blocks within the region. We expect most allocations to be long-lived and freed all at once when the region is destroyed, so we don't need to support arbitrary deallocation patterns.
 */
class IMemorySpace
{
public:
    virtual ~IMemorySpace() = default;

    virtual size_t get_total_size() const = 0;
    virtual void* getMemory() = 0;
    virtual void* get_offload_memory() = 0;
    virtual void copy_staging_to_offload(void* offload_dst, const void* staging_src, size_t bytes) = 0;
    virtual void copy_offload_to_staging(void* staging_dst, const void* offload_src, size_t bytes) = 0;
    virtual void copy_offload_to_offload(void* offload_dst, const void* offload_src, size_t bytes) = 0;
    virtual void zero_offload(void* offload_dst, size_t bytes) = 0;

    // Allocation methods for sub-allocation from memory pools
    virtual void* allocate_staging(size_t bytes) = 0;
    virtual void* allocate_offload(size_t bytes) = 0;
    virtual void release_staging(void* ptr, size_t bytes) = 0;
    virtual void release_offload(void* ptr, size_t bytes) = 0;
    virtual void reset() = 0;

    /** returns the memory space for a given offload engine, for example,
     * for Vulkan offload this would return a VulkanMemorySpace that allocates
     * from the appropriate Vulkan heap and manages staging buffers as needed.
     * For CPU-only builds this can return a simple HostMemorySpace that wraps standard heap allocation.
     */
    static IMemorySpace* get_instance();
    static IMemorySpace* get_instance(IMemorySpaceType type);
};

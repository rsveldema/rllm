#pragma once
#include <string_view>

#include <cstddef>
#include <cstdlib>
#include <cstring>

#define RLLM_DEVICE_POINTER_HAS_OFFLOAD 0

enum class IMemorySpaceType
{
    HOST,
    VULKAN
};


class OffloadMemoryBuffer
{
  public:
    OffloadMemoryBuffer() = default;

    void invalidate()
    {
        m_ptr = nullptr;
    }

    bool is_valid() const
    {
        return m_ptr != nullptr;
    }

    bool is_invalid() const
    {
        return !is_valid();
    }

    void *get() const { return m_ptr; }

  private:
    void* m_ptr = nullptr;
};

class OnHostStagingBuffer
{
  public:
    explicit OnHostStagingBuffer(void* p)
    {
        m_data = p;
    }

    void* get()
    {
        return m_data;
    }
    void* get() const
    {
        return m_data;
    }

    bool is_valid() const
    {
        return m_data != nullptr;
    }
    bool is_invalid() const
    {
        return !is_valid();
    }

    void invalidate()
    {
        m_data = nullptr;
    }

  private:
    void* m_data = nullptr;
};

/** Allocate/deallocate memory in a specific memory space.
 *
 * For CPU-only builds this is a no-op wrapper around standard heap allocation.
 * Vulkan offload uses kernel_compiler VHost/VDeviceBuffer directly.
 *
 * This will at bootup allocate a large block of memory for the entire model and training state, and then sub-allocate from that for individual tensors. This is more efficient than doing many small device allocations, and also allows us to implement a simple free list to reuse freed blocks within the region. We expect most allocations to be long-lived and freed all at once when the region is destroyed, so we don't need to support arbitrary deallocation patterns.
 */
class IMemorySpace
{
  public:
    virtual ~IMemorySpace() = default;

    virtual void copy_staging_to_offload(const OffloadMemoryBuffer& offload_dst, size_t dst_offset, const OnHostStagingBuffer& staging_src, size_t src_offset, size_t bytes, std::string_view site = {}, std::string_view parameter = {}) = 0;
    virtual void copy_offload_to_staging(const OnHostStagingBuffer& staging_dst, size_t dst_offset, const OffloadMemoryBuffer& offload_src, size_t src_offset, size_t bytes, std::string_view site = {}, std::string_view parameter = {}) = 0;
    virtual void copy_offload_to_offload(const OffloadMemoryBuffer& offload_dst, size_t dst_offset, const OffloadMemoryBuffer& offload_src, size_t src_offset, size_t bytes) = 0;
    virtual void zero_offload(const OffloadMemoryBuffer& offload_dst, size_t offset, size_t bytes) = 0;


    virtual OnHostStagingBuffer allocate_staging(size_t bytes) = 0;
    virtual OffloadMemoryBuffer allocate_offload(size_t bytes) = 0;

    virtual void release_staging(OnHostStagingBuffer& ref) = 0;
    virtual void release_offload(OffloadMemoryBuffer& ref) = 0;


    /** returns the memory space for a given offload engine, for example,
     * For CPU-only builds this can return a simple HostMemorySpace that wraps standard heap allocation.
     */
    static IMemorySpace& get_instance();
};

#pragma once

#include <cstddef>
#include <string_view>
#include <cstdint>
#include <mutex>
#include <vector>

#include <IMemorySpace.hpp>
#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

class VulkanMemorySpace final : public IMemorySpace
{
public:
    explicit VulkanMemorySpace();
    ~VulkanMemorySpace() override = default;

    void copy_staging_to_offload(const OffloadMemoryBuffer& offload_dst, size_t dst_offset, const OnHostStagingBuffer& staging_src, size_t src_offset, size_t bytes, std::string_view site = {}, std::string_view parameter = {}) override;
    void copy_offload_to_staging(const OnHostStagingBuffer& staging_dst, size_t dst_offset, const OffloadMemoryBuffer& offload_src, size_t src_offset, size_t bytes, std::string_view site = {}, std::string_view parameter = {}) override;
    void copy_offload_to_offload(const OffloadMemoryBuffer& offload_dst, size_t dst_offset, const OffloadMemoryBuffer& offload_src, size_t src_offset, size_t bytes) override;
    void zero_offload(const OffloadMemoryBuffer& offload_dst, size_t offset, size_t bytes) override;
    OnHostStagingBuffer allocate_staging(size_t bytes) override;
    OffloadMemoryBuffer allocate_offload(size_t bytes) override;
    void release_staging(OnHostStagingBuffer& ref) override;
    void release_offload(OffloadMemoryBuffer& ref) override;


    // Compute queue synchronization (protects vkQueueSubmit, command pool)
    void compute_lock()     { m_sync_mutex.lock(); }
    void compute_unlock()   { m_sync_mutex.unlock(); }
    std::mutex& sync_mutex() const { return m_sync_mutex; }

    // VMA memory allocation (protects vmaCreateBuffer, vmaDestroyBuffer)
    void alloc_lock()       { m_alloc_mutex.lock(); }
    void alloc_unlock()     { m_alloc_mutex.unlock(); }
    std::mutex& alloc_mutex() const { return m_alloc_mutex; }

    VkInstance instance() const { return m_instance; }
    VkPhysicalDevice physical_device() const { return m_physical_device; }
    VkDevice device() const { return m_device; }
    VkQueue queue() const { return m_queue;}
    uint32_t queue_family_index() const { return m_queue_family_index; }
    VkCommandPool command_pool() const { return m_command_pool; }

    VkCommandBuffer begin_one_time_command();
    void end_one_time_command(VkCommandBuffer cmd);

    static VulkanMemorySpace& get_instance();

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    uint32_t m_queue_family_index = 0;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;

    // Compute queue synchronization (queue submission, command pool access)
    mutable std::mutex m_sync_mutex;

    // VMA memory allocation synchronization
    mutable std::mutex m_alloc_mutex;

    void initialize_runtime();
};

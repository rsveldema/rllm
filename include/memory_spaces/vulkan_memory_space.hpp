#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <IMemorySpace.hpp>
#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

class VulkanMemorySpace final : public IMemorySpace
{
public:
    explicit VulkanMemorySpace();
    ~VulkanMemorySpace() override;

    void copy_staging_to_offload(const OffloadMemoryBuffer& offload_dst, const OnHostStagingBuffer& staging_src, size_t bytes) override;
    void copy_offload_to_staging(const OnHostStagingBuffer& staging_dst, const OffloadMemoryBuffer& offload_src, size_t bytes) override;
    void copy_offload_to_offload(const OffloadMemoryBuffer& offload_dst, const OffloadMemoryBuffer& offload_src, size_t bytes) override;
    void zero_offload(const OffloadMemoryBuffer& offload_dst, size_t bytes) override;
    void release_staging(OnHostStagingBuffer& ref) override;
    void release_offload(OffloadMemoryBuffer& ref) override;
    

    VkInstance instance() const { return m_instance; }
    VkPhysicalDevice physical_device() const { return m_physical_device; }
    VkDevice device() const { return m_device; }
    VkQueue queue() const { return m_queue;}
    uint32_t queue_family_index() const { return m_queue_family_index; }
    VkCommandPool command_pool() const { return m_command_pool; }

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    uint32_t m_queue_family_index override;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
};

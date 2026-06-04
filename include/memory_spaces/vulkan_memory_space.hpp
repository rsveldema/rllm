#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <IMemorySpace.hpp>
#include <vulkan/vulkan.h>

struct VmaAllocator_T;
struct VmaAllocation_T;
using VmaAllocator = VmaAllocator_T*;
using VmaAllocation = VmaAllocation_T*;

class VulkanMemorySpace final : public IMemorySpace
{
public:
    static constexpr size_t DEFAULT_POOL_BYTES = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    static constexpr const char* POOL_BYTES_ENV = "RLLM_VULKAN_POOL_BYTES";

    struct TransferContext
    {
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = nullptr;
        VkQueue queue = VK_NULL_HANDLE;
        uint32_t queue_family_index = 0;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VkBuffer offload_buffer = VK_NULL_HANDLE;
        const void* staging_base = nullptr;
        const void* offload_base = nullptr;
        size_t staging_size = 0;
        size_t offload_size = 0;
    };

    explicit VulkanMemorySpace(size_t pool_bytes = DEFAULT_POOL_BYTES);
    ~VulkanMemorySpace() override;

    size_t get_total_size() const override;
    void* getMemory() override;
    void* get_offload_memory() override;

    void copy_staging_to_offload(void* offload_dst, const void* staging_src, size_t bytes) override;
    void copy_offload_to_staging(void* staging_dst, const void* offload_src, size_t bytes) override;
    void copy_offload_to_offload(void* offload_dst, const void* offload_src, size_t bytes) override;
    void zero_offload(void* offload_dst, size_t bytes) override;

    VkInstance instance() const;
    VkPhysicalDevice physical_device() const;
    VkDevice device() const;
    VkQueue queue() const;
    uint32_t queue_family_index() const;
    VkCommandPool command_pool() const;
    VkBuffer offload_buffer() const;
    const void* offload_base() const;
    size_t offload_size() const;

    static void set_transfer_context(const TransferContext& ctx);

private:
    static TransferContext& transfer_context();
    static bool resolve_offset(const void* base, size_t total_size, const void* ptr, size_t bytes, VkDeviceSize& out_offset);
    static bool submit_buffer_copy(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize src_offset, VkDeviceSize dst_offset, size_t bytes);
    static bool submit_buffer_fill(VkBuffer buffer, VkDeviceSize offset, size_t bytes, uint32_t value);
    static void copy_vulkan_upload(void* offload_dst, const void* staging_src, size_t bytes);
    static void copy_vulkan_download(void* staging_dst, const void* offload_src, size_t bytes);
    static void copy_vulkan_device_copy(void* offload_dst, const void* offload_src, size_t bytes);
    static void zero_vulkan_offload(void* offload_dst, size_t bytes);
    void initialize_runtime();

    size_t pool_bytes_ = 0;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t queue_family_index_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = nullptr;
    VkBuffer staging_buffer_ = VK_NULL_HANDLE;
    VmaAllocation staging_allocation_ = nullptr;
    void* mapped_staging_base_ = nullptr;
    VkBuffer offload_buffer_ = VK_NULL_HANDLE;
    VmaAllocation offload_allocation_ = nullptr;
    std::vector<std::byte> offload_storage_;
};

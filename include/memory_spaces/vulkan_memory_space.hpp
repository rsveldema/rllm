#pragma once

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <IMemorySpace.hpp>
#include <logging.hpp>
#include <vulkan/vulkan.h>

class VulkanMemorySpace final : public IMemorySpace
{
public:
    static constexpr size_t DEFAULT_POOL_BYTES = 8ULL * 1024ULL * 1024ULL * 1024ULL;

    struct TransferContext
    {
        VkDevice device = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        uint32_t queue_family_index = 0;
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VkBuffer offload_buffer = VK_NULL_HANDLE;
        const void* staging_base = nullptr;
        const void* offload_base = nullptr;
        size_t staging_size = 0;
        size_t offload_size = 0;
    };

    explicit VulkanMemorySpace(size_t pool_bytes = DEFAULT_POOL_BYTES)
        : staging_storage_(pool_bytes)
        , offload_storage_(pool_bytes)
    {}

    size_t get_total_size() const override
    {
        return staging_storage_.size();
    }

    void* getMemory() override
    {
        return staging_storage_.data();
    }

    void* get_offload_memory() override
    {
        return offload_storage_.data();
    }

    void copy_staging_to_offload(void* offload_dst, const void* staging_src, size_t bytes) override
    {
        if (bytes == 0)
            return;
        copy_vulkan_upload(offload_dst, staging_src, bytes);
    }

    void copy_offload_to_staging(void* staging_dst, const void* offload_src, size_t bytes) override
    {
        if (bytes == 0)
            return;
        copy_vulkan_download(staging_dst, offload_src, bytes);
    }

    void copy_offload_to_offload(void* offload_dst, const void* offload_src, size_t bytes) override
    {
        if (bytes == 0)
            return;
        copy_vulkan_device_copy(offload_dst, offload_src, bytes);
    }

    void zero_offload(void* offload_dst, size_t bytes) override
    {
        if (bytes == 0)
            return;
        zero_vulkan_offload(offload_dst, bytes);
    }

    static void set_transfer_context(const TransferContext& ctx)
    {
        transfer_context() = ctx;
    }

private:
    static TransferContext& transfer_context()
    {
        static TransferContext callback;
        return callback;
    }

    static bool resolve_offset(const void* base, size_t total_size, const void* ptr, size_t bytes, VkDeviceSize& out_offset)
    {
        if (base == nullptr || ptr == nullptr || bytes > total_size)
            return false;

        const auto* base_bytes = static_cast<const std::byte*>(base);
        const auto* ptr_bytes = static_cast<const std::byte*>(ptr);

        if (ptr_bytes < base_bytes)
            return false;

        const size_t offset = static_cast<size_t>(ptr_bytes - base_bytes);
        if (offset > total_size || bytes > (total_size - offset))
            return false;

        out_offset = static_cast<VkDeviceSize>(offset);
        return true;
    }

    static bool submit_buffer_copy(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize src_offset, VkDeviceSize dst_offset, size_t bytes)
    {
        TransferContext& ctx = transfer_context();
        if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE)
        {
            LOG_ERROR("Vulkan transfer context is invalid (device={} queue={})", static_cast<void*>(ctx.device), static_cast<void*>(ctx.queue));
            return false;
        }

        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = ctx.queue_family_index;

        const VkResult create_pool_result = vkCreateCommandPool(ctx.device, &pool_info, nullptr, &command_pool);
        if (create_pool_result != VK_SUCCESS)
        {
            LOG_ERROR("vkCreateCommandPool failed: {}", static_cast<int>(create_pool_result));
            return false;
        }

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        bool success = false;
        const VkResult alloc_result = vkAllocateCommandBuffers(ctx.device, &alloc_info, &command_buffer);
        if (alloc_result == VK_SUCCESS)
        {
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            const VkResult begin_result = vkBeginCommandBuffer(command_buffer, &begin_info);
            if (begin_result == VK_SUCCESS)
            {
                VkBufferCopy region{};
                region.srcOffset = src_offset;
                region.dstOffset = dst_offset;
                region.size = static_cast<VkDeviceSize>(bytes);
                vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &region);

                const VkResult end_result = vkEndCommandBuffer(command_buffer);
                if (end_result == VK_SUCCESS)
                {
                    VkSubmitInfo submit_info{};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &command_buffer;

                    const VkResult submit_result = vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE);
                    if (submit_result == VK_SUCCESS)
                    {
                        const VkResult wait_result = vkQueueWaitIdle(ctx.queue);
                        if (wait_result == VK_SUCCESS)
                        {
                            success = true;
                        }
                        else
                        {
                            LOG_ERROR("vkQueueWaitIdle failed: {}", static_cast<int>(wait_result));
                        }
                    }
                    else
                    {
                        LOG_ERROR("vkQueueSubmit failed: {}", static_cast<int>(submit_result));
                    }
                }
                else
                {
                    LOG_ERROR("vkEndCommandBuffer failed: {}", static_cast<int>(end_result));
                }
            }
            else
            {
                LOG_ERROR("vkBeginCommandBuffer failed: {}", static_cast<int>(begin_result));
            }
        }
        else
        {
            LOG_ERROR("vkAllocateCommandBuffers failed: {}", static_cast<int>(alloc_result));
        }

        if (command_buffer != VK_NULL_HANDLE)
            vkFreeCommandBuffers(ctx.device, command_pool, 1, &command_buffer);
        vkDestroyCommandPool(ctx.device, command_pool, nullptr);
        return success;
    }

    static bool submit_buffer_fill(VkBuffer buffer, VkDeviceSize offset, size_t bytes, uint32_t value)
    {
        TransferContext& ctx = transfer_context();
        if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE)
        {
            LOG_ERROR("Vulkan transfer context is invalid (device={} queue={})", static_cast<void*>(ctx.device), static_cast<void*>(ctx.queue));
            return false;
        }

        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = ctx.queue_family_index;

        const VkResult create_pool_result = vkCreateCommandPool(ctx.device, &pool_info, nullptr, &command_pool);
        if (create_pool_result != VK_SUCCESS)
        {
            LOG_ERROR("vkCreateCommandPool failed: {}", static_cast<int>(create_pool_result));
            return false;
        }

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        bool success = false;
        const VkResult alloc_result = vkAllocateCommandBuffers(ctx.device, &alloc_info, &command_buffer);
        if (alloc_result == VK_SUCCESS)
        {
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            const VkResult begin_result = vkBeginCommandBuffer(command_buffer, &begin_info);
            if (begin_result == VK_SUCCESS)
            {
                vkCmdFillBuffer(command_buffer, buffer, offset, static_cast<VkDeviceSize>(bytes), value);

                const VkResult end_result = vkEndCommandBuffer(command_buffer);
                if (end_result == VK_SUCCESS)
                {
                    VkSubmitInfo submit_info{};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &command_buffer;

                    const VkResult submit_result = vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE);
                    if (submit_result == VK_SUCCESS)
                    {
                        const VkResult wait_result = vkQueueWaitIdle(ctx.queue);
                        if (wait_result == VK_SUCCESS)
                        {
                            success = true;
                        }
                        else
                        {
                            LOG_ERROR("vkQueueWaitIdle failed: {}", static_cast<int>(wait_result));
                        }
                    }
                    else
                    {
                        LOG_ERROR("vkQueueSubmit failed: {}", static_cast<int>(submit_result));
                    }
                }
                else
                {
                    LOG_ERROR("vkEndCommandBuffer failed: {}", static_cast<int>(end_result));
                }
            }
            else
            {
                LOG_ERROR("vkBeginCommandBuffer failed: {}", static_cast<int>(begin_result));
            }
        }
        else
        {
            LOG_ERROR("vkAllocateCommandBuffers failed: {}", static_cast<int>(alloc_result));
        }

        if (command_buffer != VK_NULL_HANDLE)
            vkFreeCommandBuffers(ctx.device, command_pool, 1, &command_buffer);
        vkDestroyCommandPool(ctx.device, command_pool, nullptr);
        return success;
    }

    static void copy_vulkan_upload(void* offload_dst, const void* staging_src, size_t bytes)
    {
        TransferContext& ctx = transfer_context();

        // Until Vulkan transfer buffers are fully provisioned, keep host/offload mirrors coherent.
        if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE ||
            ctx.staging_buffer == VK_NULL_HANDLE || ctx.offload_buffer == VK_NULL_HANDLE ||
            ctx.staging_base == nullptr || ctx.offload_base == nullptr)
        {
            std::memcpy(offload_dst, staging_src, bytes);
            return;
        }

        VkDeviceSize src_offset = 0;
        VkDeviceSize dst_offset = 0;

        if (resolve_offset(ctx.staging_base, ctx.staging_size, staging_src, bytes, src_offset) &&
            resolve_offset(ctx.offload_base, ctx.offload_size, offload_dst, bytes, dst_offset) &&
            submit_buffer_copy(ctx.staging_buffer, ctx.offload_buffer, src_offset, dst_offset, bytes))
        {
            return;
        }
        LOG_ERROR("Fatal: Vulkan offload upload failed and fallback is disabled.");
        std::abort();
    }

    static void copy_vulkan_download(void* staging_dst, const void* offload_src, size_t bytes)
    {
        TransferContext& ctx = transfer_context();

        // Until Vulkan transfer buffers are fully provisioned, keep host/offload mirrors coherent.
        if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE ||
            ctx.staging_buffer == VK_NULL_HANDLE || ctx.offload_buffer == VK_NULL_HANDLE ||
            ctx.staging_base == nullptr || ctx.offload_base == nullptr)
        {
            std::memcpy(staging_dst, offload_src, bytes);
            return;
        }

        VkDeviceSize src_offset = 0;
        VkDeviceSize dst_offset = 0;

        if (resolve_offset(ctx.offload_base, ctx.offload_size, offload_src, bytes, src_offset) &&
            resolve_offset(ctx.staging_base, ctx.staging_size, staging_dst, bytes, dst_offset) &&
            submit_buffer_copy(ctx.offload_buffer, ctx.staging_buffer, src_offset, dst_offset, bytes))
        {
            return;
        }
        LOG_ERROR("Fatal: Vulkan offload download failed and fallback is disabled.");
        std::abort();
    }

    static void copy_vulkan_device_copy(void* offload_dst, const void* offload_src, size_t bytes)
    {
        TransferContext& ctx = transfer_context();

        if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE ||
            ctx.offload_buffer == VK_NULL_HANDLE || ctx.offload_base == nullptr)
        {
            std::memcpy(offload_dst, offload_src, bytes);
            return;
        }

        VkDeviceSize src_offset = 0;
        VkDeviceSize dst_offset = 0;

        if (resolve_offset(ctx.offload_base, ctx.offload_size, offload_src, bytes, src_offset) &&
            resolve_offset(ctx.offload_base, ctx.offload_size, offload_dst, bytes, dst_offset) &&
            submit_buffer_copy(ctx.offload_buffer, ctx.offload_buffer, src_offset, dst_offset, bytes))
        {
            return;
        }

        LOG_ERROR("Fatal: Vulkan offload device copy failed and fallback is disabled.");
        std::abort();
    }

    static void zero_vulkan_offload(void* offload_dst, size_t bytes)
    {
        TransferContext& ctx = transfer_context();

        if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE ||
            ctx.offload_buffer == VK_NULL_HANDLE || ctx.offload_base == nullptr)
        {
            std::memset(offload_dst, 0, bytes);
            return;
        }

        VkDeviceSize dst_offset = 0;
        if ((bytes % 4u) == 0u &&
            resolve_offset(ctx.offload_base, ctx.offload_size, offload_dst, bytes, dst_offset) &&
            submit_buffer_fill(ctx.offload_buffer, dst_offset, bytes, 0u))
        {
            return;
        }

        LOG_ERROR("Fatal: Vulkan offload zero failed and fallback is disabled.");
        std::abort();
    }

    std::vector<std::byte> staging_storage_;
    std::vector<std::byte> offload_storage_;
};

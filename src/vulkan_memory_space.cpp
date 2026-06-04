#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <memory_spaces/vulkan_memory_space.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <logging.hpp>

namespace
{
    struct VulkanCandidate
    {
        VkPhysicalDevice device = VK_NULL_HANDLE;
        uint32_t queue_family_index = 0;
        VkPhysicalDeviceProperties props{};
    };

    std::string to_lower_copy(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    uint32_t vendor_id_from_name(const std::string& name)
    {
        const std::string lowered = to_lower_copy(name);
        if (lowered == "nvidia") return 0x10DE;
        if (lowered == "amd") return 0x1002;
        if (lowered == "intel") return 0x8086;
        if (lowered == "arm") return 0x13B5;
        if (lowered == "qualcomm") return 0x5143;
        if (lowered == "apple") return 0x106B;
        if (lowered == "google") return 0x1AE0;
        if (lowered == "mesa") return 0x10005;
        return 0;
    }

    bool contains_case_insensitive(const std::string& haystack, const std::string& needle)
    {
        if (needle.empty())
            return true;
        return to_lower_copy(haystack).find(to_lower_copy(needle)) != std::string::npos;
    }

    int device_score(const VulkanCandidate& c)
    {
        int score = 0;
        if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 500;
        if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) score += 250;
        const std::string name = to_lower_copy(c.props.deviceName);
        if (name.find("llvmpipe") != std::string::npos || name.find("lavapipe") != std::string::npos || name.find("software") != std::string::npos)
            score -= 10000;
        return score;
    }

    const VulkanCandidate* pick_candidate(const std::vector<VulkanCandidate>& candidates)
    {
        if (candidates.empty())
            return nullptr;

        const char* index_env = std::getenv("RLLM_VULKAN_DEVICE_INDEX");
        if (index_env != nullptr && *index_env != '\0')
        {
            const long idx = std::strtol(index_env, nullptr, 10);
            if (idx >= 0 && static_cast<size_t>(idx) < candidates.size())
                return &candidates[static_cast<size_t>(idx)];
        }

        const char* vendor_env = std::getenv("RLLM_VULKAN_VENDOR");
        const char* name_env = std::getenv("RLLM_VULKAN_DEVICE_SUBSTRING");
        const uint32_t vendor_id = vendor_env ? vendor_id_from_name(vendor_env) : 0u;
        const std::string name_substring = name_env ? std::string(name_env) : std::string();

        if (vendor_id != 0u || !name_substring.empty())
        {
            for (const VulkanCandidate& c : candidates)
            {
                if (vendor_id != 0u && c.props.vendorID != vendor_id)
                    continue;
                if (!contains_case_insensitive(c.props.deviceName, name_substring))
                    continue;
                return &c;
            }
        }

        return &*std::max_element(candidates.begin(), candidates.end(), [](const VulkanCandidate& a, const VulkanCandidate& b) {
            return device_score(a) < device_score(b);
        });
    }

    bool create_transfer_staging_buffer(VmaAllocator allocator, size_t bytes, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, void*& mapped)
    {
        mapped = nullptr;

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = static_cast<VkDeviceSize>(bytes);
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo allocation_info{};
        if (vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer, &allocation, &allocation_info) != VK_SUCCESS)
            return false;

        mapped = allocation_info.pMappedData;
        if (mapped == nullptr && vmaMapMemory(allocator, allocation, &mapped) != VK_SUCCESS)
        {
            vmaDestroyBuffer(allocator, buffer, allocation);
            buffer = VK_NULL_HANDLE;
            allocation = nullptr;
            return false;
        }

        return true;
    }

    size_t effective_pool_bytes(size_t default_pool_bytes)
    {
        const char* value = std::getenv(VulkanMemorySpace::POOL_BYTES_ENV);
        if (value == nullptr || *value == '\0')
            return default_pool_bytes;

        errno = 0;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
            parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
        {
            LOG_ERROR(
                "Invalid {}='{}'. Expected a positive byte count.",
                VulkanMemorySpace::POOL_BYTES_ENV,
                value
            );
            std::abort();
        }

        return static_cast<size_t>(parsed);
    }
}

VulkanMemorySpace::VulkanMemorySpace(size_t pool_bytes)
    : pool_bytes_(effective_pool_bytes(pool_bytes))
    , offload_storage_(pool_bytes_)
{
    initialize_runtime();
}

VulkanMemorySpace::~VulkanMemorySpace()
{
    transfer_context() = TransferContext{};

    if (mapped_staging_base_ != nullptr && allocator_ != nullptr && staging_allocation_ != nullptr)
    {
        vmaUnmapMemory(allocator_, staging_allocation_);
        mapped_staging_base_ = nullptr;
    }
    if (allocator_ != nullptr && offload_buffer_ != VK_NULL_HANDLE && offload_allocation_ != nullptr)
        vmaDestroyBuffer(allocator_, offload_buffer_, offload_allocation_);
    if (allocator_ != nullptr && staging_buffer_ != VK_NULL_HANDLE && staging_allocation_ != nullptr)
        vmaDestroyBuffer(allocator_, staging_buffer_, staging_allocation_);
    if (allocator_ != nullptr)
        vmaDestroyAllocator(allocator_);
    if (command_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);
    if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);
}

size_t VulkanMemorySpace::get_total_size() const
{
    return pool_bytes_;
}

void* VulkanMemorySpace::getMemory()
{
    return mapped_staging_base_;
}

void* VulkanMemorySpace::get_offload_memory()
{
    return offload_storage_.data();
}

void VulkanMemorySpace::copy_staging_to_offload(void* offload_dst, const void* staging_src, size_t bytes)
{
    assert(bytes != 0);
    copy_vulkan_upload(offload_dst, staging_src, bytes);
}

void VulkanMemorySpace::copy_offload_to_staging(void* staging_dst, const void* offload_src, size_t bytes)
{
    assert(bytes != 0);
    copy_vulkan_download(staging_dst, offload_src, bytes);
}

void VulkanMemorySpace::copy_offload_to_offload(void* offload_dst, const void* offload_src, size_t bytes)
{
    assert(bytes != 0);
    copy_vulkan_device_copy(offload_dst, offload_src, bytes);
}

void VulkanMemorySpace::zero_offload(void* offload_dst, size_t bytes)
{
    assert(bytes != 0);
    zero_vulkan_offload(offload_dst, bytes);
}

VkInstance VulkanMemorySpace::instance() const
{
    return instance_;
}

VkPhysicalDevice VulkanMemorySpace::physical_device() const
{
    return physical_device_;
}

VkDevice VulkanMemorySpace::device() const
{
    return device_;
}

VkQueue VulkanMemorySpace::queue() const
{
    return queue_;
}

uint32_t VulkanMemorySpace::queue_family_index() const
{
    return queue_family_index_;
}

VkCommandPool VulkanMemorySpace::command_pool() const
{
    return command_pool_;
}

VkBuffer VulkanMemorySpace::offload_buffer() const
{
    return offload_buffer_;
}

const void* VulkanMemorySpace::offload_base() const
{
    return offload_storage_.data();
}

size_t VulkanMemorySpace::offload_size() const
{
    return offload_storage_.size();
}

void VulkanMemorySpace::set_transfer_context(const TransferContext& ctx)
{
    transfer_context() = ctx;
}

VulkanMemorySpace::TransferContext& VulkanMemorySpace::transfer_context()
{
    static TransferContext callback;
    return callback;
}

bool VulkanMemorySpace::resolve_offset(const void* base, size_t total_size, const void* ptr, size_t bytes, VkDeviceSize& out_offset)
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

bool VulkanMemorySpace::submit_buffer_copy(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize src_offset, VkDeviceSize dst_offset, size_t bytes)
{
    TransferContext& ctx = transfer_context();
    if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE)
    {
        LOG_ERROR("Vulkan transfer context is invalid.");
        return false;
    }

    VkCommandPool transient_pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = ctx.queue_family_index;
    const VkResult create_pool_result = vkCreateCommandPool(ctx.device, &pool_info, nullptr, &transient_pool);
    if (create_pool_result != VK_SUCCESS)
    {
        LOG_ERROR("vkCreateCommandPool failed for transfer copy: {}", static_cast<int>(create_pool_result));
        return false;
    }

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = transient_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    const VkResult alloc_result = vkAllocateCommandBuffers(ctx.device, &alloc_info, &command_buffer);
    if (alloc_result != VK_SUCCESS)
    {
        LOG_ERROR("vkAllocateCommandBuffers failed: {}", static_cast<int>(alloc_result));
        vkDestroyCommandPool(ctx.device, transient_pool, nullptr);
        return false;
    }

    bool success = false;
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
            const VkResult wait_result = submit_result == VK_SUCCESS ? vkQueueWaitIdle(ctx.queue) : VK_NOT_READY;
            success = submit_result == VK_SUCCESS && wait_result == VK_SUCCESS;
            if (!success)
            {
                LOG_ERROR(
                    "Transfer copy failed: submit={} wait={} src_offset={} dst_offset={} bytes={}",
                    static_cast<int>(submit_result),
                    static_cast<int>(wait_result),
                    static_cast<unsigned long long>(src_offset),
                    static_cast<unsigned long long>(dst_offset),
                    bytes
                );
            }
        }
        else
        {
            LOG_ERROR("vkEndCommandBuffer failed for transfer copy: {}", static_cast<int>(end_result));
        }
    }
    else
    {
        LOG_ERROR("vkBeginCommandBuffer failed for transfer copy: {}", static_cast<int>(begin_result));
    }

    vkFreeCommandBuffers(ctx.device, transient_pool, 1, &command_buffer);
    vkDestroyCommandPool(ctx.device, transient_pool, nullptr);
    return success;
}

bool VulkanMemorySpace::submit_buffer_fill(VkBuffer buffer, VkDeviceSize offset, size_t bytes, uint32_t value)
{
    TransferContext& ctx = transfer_context();
    if (ctx.device == VK_NULL_HANDLE || ctx.queue == VK_NULL_HANDLE)
    {
        LOG_ERROR("Vulkan transfer context is invalid.");
        return false;
    }

    VkCommandPool transient_pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = ctx.queue_family_index;
    const VkResult create_pool_result = vkCreateCommandPool(ctx.device, &pool_info, nullptr, &transient_pool);
    if (create_pool_result != VK_SUCCESS)
    {
        LOG_ERROR("vkCreateCommandPool failed for transfer fill: {}", static_cast<int>(create_pool_result));
        return false;
    }

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = transient_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    const VkResult alloc_result = vkAllocateCommandBuffers(ctx.device, &alloc_info, &command_buffer);
    if (alloc_result != VK_SUCCESS)
    {
        LOG_ERROR("vkAllocateCommandBuffers failed: {}", static_cast<int>(alloc_result));
        vkDestroyCommandPool(ctx.device, transient_pool, nullptr);
        return false;
    }

    bool success = false;
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
            const VkResult wait_result = submit_result == VK_SUCCESS ? vkQueueWaitIdle(ctx.queue) : VK_NOT_READY;
            success = submit_result == VK_SUCCESS && wait_result == VK_SUCCESS;
            if (!success)
            {
                LOG_ERROR(
                    "Transfer fill failed: submit={} wait={} offset={} bytes={}",
                    static_cast<int>(submit_result),
                    static_cast<int>(wait_result),
                    static_cast<unsigned long long>(offset),
                    bytes
                );
            }
        }
        else
        {
            LOG_ERROR("vkEndCommandBuffer failed for transfer fill: {}", static_cast<int>(end_result));
        }
    }
    else
    {
        LOG_ERROR("vkBeginCommandBuffer failed for transfer fill: {}", static_cast<int>(begin_result));
    }

    vkFreeCommandBuffers(ctx.device, transient_pool, 1, &command_buffer);
    vkDestroyCommandPool(ctx.device, transient_pool, nullptr);
    return success;
}

void VulkanMemorySpace::copy_vulkan_upload(void* offload_dst, const void* staging_src, size_t bytes)
{
    TransferContext& ctx = transfer_context();
    VkDeviceSize src_offset = 0;
    VkDeviceSize dst_offset = 0;
    const bool src_ok = resolve_offset(ctx.staging_base, ctx.staging_size, staging_src, bytes, src_offset);
    const bool dst_ok = resolve_offset(ctx.offload_base, ctx.offload_size, offload_dst, bytes, dst_offset);
    if (src_ok && dst_ok && submit_buffer_copy(ctx.staging_buffer, ctx.offload_buffer, src_offset, dst_offset, bytes))
    {
        return;
    }

    if (!src_ok && dst_ok && ctx.allocator != nullptr)
    {
        VkBuffer temp_buffer = VK_NULL_HANDLE;
        VmaAllocation temp_allocation = nullptr;
        void* mapped = nullptr;
        if (create_transfer_staging_buffer(ctx.allocator, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, temp_buffer, temp_allocation, mapped))
        {
            std::memcpy(mapped, staging_src, bytes);
            if (submit_buffer_copy(temp_buffer, ctx.offload_buffer, 0, dst_offset, bytes))
            {
                vmaDestroyBuffer(ctx.allocator, temp_buffer, temp_allocation);
                return;
            }
            vmaDestroyBuffer(ctx.allocator, temp_buffer, temp_allocation);
        }
    }

    LOG_ERROR(
        "Fatal: Vulkan offload upload failed and fallback is disabled (src_ok={} dst_ok={} bytes={} staging_base={} staging_src={} offload_base={} offload_dst={}).",
        src_ok,
        dst_ok,
        bytes,
        ctx.staging_base,
        staging_src,
        ctx.offload_base,
        offload_dst
    );
    std::abort();
}

void VulkanMemorySpace::copy_vulkan_download(void* staging_dst, const void* offload_src, size_t bytes)
{
    TransferContext& ctx = transfer_context();
    VkDeviceSize src_offset = 0;
    VkDeviceSize dst_offset = 0;
    const bool src_ok = resolve_offset(ctx.offload_base, ctx.offload_size, offload_src, bytes, src_offset);
    const bool dst_ok = resolve_offset(ctx.staging_base, ctx.staging_size, staging_dst, bytes, dst_offset);
    if (src_ok && dst_ok && submit_buffer_copy(ctx.offload_buffer, ctx.staging_buffer, src_offset, dst_offset, bytes))
    {
        return;
    }

    if (src_ok && !dst_ok && ctx.allocator != nullptr)
    {
        VkBuffer temp_buffer = VK_NULL_HANDLE;
        VmaAllocation temp_allocation = nullptr;
        void* mapped = nullptr;
        if (create_transfer_staging_buffer(ctx.allocator, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, temp_buffer, temp_allocation, mapped))
        {
            if (submit_buffer_copy(ctx.offload_buffer, temp_buffer, src_offset, 0, bytes))
            {
                std::memcpy(staging_dst, mapped, bytes);
                vmaDestroyBuffer(ctx.allocator, temp_buffer, temp_allocation);
                return;
            }
            vmaDestroyBuffer(ctx.allocator, temp_buffer, temp_allocation);
        }
    }

    LOG_ERROR("Fatal: Vulkan offload download failed and fallback is disabled.");
    std::abort();
}

void VulkanMemorySpace::copy_vulkan_device_copy(void* offload_dst, const void* offload_src, size_t bytes)
{
    TransferContext& ctx = transfer_context();
    VkDeviceSize src_offset = 0;
    VkDeviceSize dst_offset = 0;
    if (resolve_offset(ctx.offload_base, ctx.offload_size, offload_src, bytes, src_offset)
        && resolve_offset(ctx.offload_base, ctx.offload_size, offload_dst, bytes, dst_offset)
        && submit_buffer_copy(ctx.offload_buffer, ctx.offload_buffer, src_offset, dst_offset, bytes))
    {
        return;
    }

    LOG_ERROR("Fatal: Vulkan offload device copy failed and fallback is disabled.");
    std::abort();
}

void VulkanMemorySpace::zero_vulkan_offload(void* offload_dst, size_t bytes)
{
    TransferContext& ctx = transfer_context();
    VkDeviceSize dst_offset = 0;
    if ((bytes % 4u) == 0u
        && resolve_offset(ctx.offload_base, ctx.offload_size, offload_dst, bytes, dst_offset)
        && submit_buffer_fill(ctx.offload_buffer, dst_offset, bytes, 0u))
    {
        return;
    }

    LOG_ERROR("Fatal: Vulkan offload zero failed and fallback is disabled.");
    std::abort();
}

void VulkanMemorySpace::initialize_runtime()
{
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "rllm";
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app_info.pEngineName = "rllm";
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    if (vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS)
    {
        LOG_ERROR("vkCreateInstance failed.");
        std::abort();
    }

    uint32_t physical_count = 0;
    if (vkEnumeratePhysicalDevices(instance_, &physical_count, nullptr) != VK_SUCCESS || physical_count == 0)
    {
        LOG_ERROR("vkEnumeratePhysicalDevices failed or found no devices.");
        std::abort();
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    if (vkEnumeratePhysicalDevices(instance_, &physical_count, physical_devices.data()) != VK_SUCCESS)
    {
        LOG_ERROR("vkEnumeratePhysicalDevices (fill) failed.");
        std::abort();
    }

    std::vector<VulkanCandidate> candidates;
    for (const VkPhysicalDevice physical_device : physical_devices)
    {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_props(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_props.data());
        for (uint32_t i = 0; i < queue_family_count; ++i)
        {
            if ((queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u)
            {
                VulkanCandidate candidate{};
                candidate.device = physical_device;
                candidate.queue_family_index = i;
                vkGetPhysicalDeviceProperties(physical_device, &candidate.props);
                candidates.push_back(candidate);
                break;
            }
        }
    }

    const VulkanCandidate* chosen = pick_candidate(candidates);
    if (chosen == nullptr)
    {
        LOG_ERROR("No Vulkan physical device with a compute-capable queue family was found.");
        std::abort();
    }

    physical_device_ = chosen->device;
    queue_family_index_ = chosen->queue_family_index;

    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT supported_atomic_float2{};
    supported_atomic_float2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supported_atomic_float{};
    supported_atomic_float.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    supported_atomic_float.pNext = &supported_atomic_float2;
    VkPhysicalDeviceFeatures2 supported_features{};
    supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supported_features.pNext = &supported_atomic_float;
    vkGetPhysicalDeviceFeatures2(physical_device_, &supported_features);

    if (!supported_atomic_float.shaderBufferFloat32AtomicAdd
        || !supported_atomic_float.shaderSharedFloat32AtomicAdd
        || !supported_atomic_float2.shaderBufferFloat32AtomicMinMax
        || !supported_atomic_float2.shaderSharedFloat32AtomicMinMax)
    {
        LOG_ERROR("Selected Vulkan device does not support required float atomic features.");
        std::abort();
    }

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family_index_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT atomic_float2_features{};
    atomic_float2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
    atomic_float2_features.shaderBufferFloat32AtomicMinMax = VK_TRUE;
    atomic_float2_features.shaderSharedFloat32AtomicMinMax = VK_TRUE;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
    atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    atomic_float_features.pNext = &atomic_float2_features;
    atomic_float_features.shaderBufferFloat32AtomicAdd = VK_TRUE;
    atomic_float_features.shaderSharedFloat32AtomicAdd = VK_TRUE;

    const char* device_extensions[] = {
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME,
    };

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 2;
    device_info.ppEnabledExtensionNames = device_extensions;
    device_info.pNext = &atomic_float_features;
    if (vkCreateDevice(physical_device_, &device_info, nullptr, &device_) != VK_SUCCESS)
    {
        LOG_ERROR("vkCreateDevice failed.");
        std::abort();
    }

    vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index_;
    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS)
    {
        LOG_ERROR("vkCreateCommandPool failed.");
        std::abort();
    }

    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.instance = instance_;
    allocator_info.physicalDevice = physical_device_;
    allocator_info.device = device_;
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_1;
    if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS)
    {
        LOG_ERROR("vmaCreateAllocator failed.");
        std::abort();
    }

    VkBufferCreateInfo staging_buffer_info{};
    staging_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_buffer_info.size = static_cast<VkDeviceSize>(pool_bytes_);
    staging_buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    staging_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo staging_alloc_info{};
    staging_alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo staging_info{};
    if (vmaCreateBuffer(allocator_, &staging_buffer_info, &staging_alloc_info, &staging_buffer_, &staging_allocation_, &staging_info) != VK_SUCCESS)
    {
        LOG_ERROR("vmaCreateBuffer failed for staging buffer.");
        std::abort();
    }
    mapped_staging_base_ = staging_info.pMappedData;
    if (mapped_staging_base_ == nullptr && vmaMapMemory(allocator_, staging_allocation_, &mapped_staging_base_) != VK_SUCCESS)
    {
        LOG_ERROR("vmaMapMemory failed for staging buffer.");
        std::abort();
    }

    VkBufferCreateInfo offload_buffer_info{};
    offload_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    offload_buffer_info.size = static_cast<VkDeviceSize>(pool_bytes_);
    offload_buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    offload_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo offload_alloc_info{};
    offload_alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(allocator_, &offload_buffer_info, &offload_alloc_info, &offload_buffer_, &offload_allocation_, nullptr) != VK_SUCCESS)
    {
        LOG_ERROR("vmaCreateBuffer failed for offload buffer.");
        std::abort();
    }

    TransferContext ctx{};
    ctx.device = device_;
    ctx.allocator = allocator_;
    ctx.queue = queue_;
    ctx.queue_family_index = queue_family_index_;
    ctx.command_pool = command_pool_;
    ctx.staging_buffer = staging_buffer_;
    ctx.offload_buffer = offload_buffer_;
    ctx.staging_base = mapped_staging_base_;
    ctx.offload_base = offload_storage_.data();
    ctx.staging_size = pool_bytes_;
    ctx.offload_size = pool_bytes_;
    set_transfer_context(ctx);
}

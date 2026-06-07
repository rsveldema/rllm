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
#include <mutex>

#include <logging.hpp>
#include <parallel.hpp>

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
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s;
    }

    uint32_t vendor_id_from_name(const std::string& name)
    {
        const std::string lowered = to_lower_copy(name);
        if (lowered == "nvidia")
            return 0x10DE;
        if (lowered == "amd")
            return 0x1002;
        if (lowered == "intel")
            return 0x8086;
        if (lowered == "arm")
            return 0x13B5;
        if (lowered == "qualcomm")
            return 0x5143;
        if (lowered == "apple")
            return 0x106B;
        if (lowered == "google")
            return 0x1AE0;
        if (lowered == "mesa")
            return 0x10005;
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
        if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            score += 1000;
        if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            score += 500;
        if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
            score += 250;
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

} // namespace


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
    const VkResult create_instance_result = vkCreateInstance(&instance_info, nullptr, &m_instance);
    if (create_instance_result != VK_SUCCESS)
    {
        LOG_ERROR("vkCreateInstance failed: result={}", static_cast<int>(create_instance_result));
        std::abort();
    }

    uint32_t physical_count = 0;
    const VkResult enumerate_count_result = vkEnumeratePhysicalDevices(m_instance, &physical_count, nullptr);
    if (enumerate_count_result != VK_SUCCESS || physical_count == 0)
    {
        LOG_ERROR("vkEnumeratePhysicalDevices failed or found no devices: result={} physical_count={}", static_cast<int>(enumerate_count_result), physical_count);
        std::abort();
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    const VkResult enumerate_fill_result = vkEnumeratePhysicalDevices(m_instance, &physical_count, physical_devices.data());
    if (enumerate_fill_result != VK_SUCCESS)
    {
        LOG_ERROR("vkEnumeratePhysicalDevices (fill) failed: result={}", static_cast<int>(enumerate_fill_result));
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
        LOG_ERROR("No Vulkan physical device with a compute-capable queue family was found. physical_count={} candidates={}", physical_count, candidates.size());
        std::abort();
    }

    m_physical_device = chosen->device;
    m_queue_family_index = chosen->queue_family_index;

    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT supported_atomic_float2{};
    supported_atomic_float2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supported_atomic_float{};
    supported_atomic_float.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    supported_atomic_float.pNext = &supported_atomic_float2;
    VkPhysicalDeviceFeatures2 supported_features{};
    supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supported_features.pNext = &supported_atomic_float;
    vkGetPhysicalDeviceFeatures2(m_physical_device, &supported_features);

    if (!supported_atomic_float.shaderBufferFloat32AtomicAdd || !supported_atomic_float.shaderSharedFloat32AtomicAdd || !supported_atomic_float2.shaderBufferFloat32AtomicMinMax || !supported_atomic_float2.shaderSharedFloat32AtomicMinMax)
    {
        LOG_ERROR(
            "Selected Vulkan device '{}' does not support required float atomic features: "
            "buffer_add={} shared_add={} buffer_minmax={} shared_minmax={}",
            chosen->props.deviceName,
            static_cast<bool>(supported_atomic_float.shaderBufferFloat32AtomicAdd),
            static_cast<bool>(supported_atomic_float.shaderSharedFloat32AtomicAdd),
            static_cast<bool>(supported_atomic_float2.shaderBufferFloat32AtomicMinMax),
            static_cast<bool>(supported_atomic_float2.shaderSharedFloat32AtomicMinMax)
        );
        std::abort();
    }

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = m_queue_family_index;
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
    const VkResult create_device_result = vkCreateDevice(m_physical_device, &device_info, nullptr, &m_device);
    if (create_device_result != VK_SUCCESS)
    {
        LOG_ERROR("vkCreateDevice failed: result={} device='{}'", static_cast<int>(create_device_result), chosen->props.deviceName);
        std::abort();
    }
    vkGetDeviceQueue(m_device, m_queue_family_index, 0, &m_queue);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = m_queue_family_index;
    const VkResult create_pool_result = vkCreateCommandPool(m_device, &pool_info, nullptr, &m_command_pool);
    if (create_pool_result != VK_SUCCESS)
    {
        LOG_ERROR("vkCreateCommandPool failed: result={} queue_family={}", static_cast<int>(create_pool_result), m_queue_family_index);
        std::abort();
    }

    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.instance = m_instance;
    allocator_info.physicalDevice = m_physical_device;
    allocator_info.device = m_device;
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_4;
    const VkResult create_allocator_result = vmaCreateAllocator(&allocator_info, &m_allocator);
    if (create_allocator_result != VK_SUCCESS)
    {
        LOG_ERROR("vmaCreateAllocator failed: result={}", static_cast<int>(create_allocator_result));
        std::abort();
    }
}

namespace
{
    void create_vulkan_buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage, VkBuffer& buffer, VmaAllocation& allocation)
    {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocation_info{};
        allocation_info.usage = memory_usage;

        const VkResult result = vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &buffer, &allocation, nullptr);
        if (result != VK_SUCCESS)
        {
            LOG_ERROR("vmaCreateBuffer failed: result={}", static_cast<int>(result));
            std::abort();
        }
    }

    void destroy_vulkan_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation)
    {
        if (buffer != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, buffer, allocation);
        }
    }
} // namespace

// ---- Copy / zero operations ---------------------------------------------------
// These call begin_one_time_command()/end_one_time_command() which each handle
// their own m_sync_mutex locking internally. Therefore the caller must NOT
// acquire m_sync_mutex first (that would cause a recursive-lock deadlock).

void VulkanMemorySpace::copy_staging_to_offload(const OffloadMemoryBuffer& offload_dst, size_t dst_offset, const OnHostStagingBuffer& staging_src, size_t src_offset, size_t bytes, std::string_view site, std::string_view parameter)
{
    assert(bytes != 0);
    assert(offload_dst.is_valid());
    assert(staging_src.is_valid());

#if defined(RLLM_ENABLE_STATISTICS)
    if (!site.empty())
        parallel::statistics.record_host_to_device_buffer_copy(site, parameter, bytes);
#endif

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = VK_NULL_HANDLE;
    create_vulkan_buffer(m_allocator, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, staging_buffer, staging_allocation);

    void* mapped = nullptr;
    vmaMapMemory(m_allocator, staging_allocation, &mapped);
    std::memcpy(static_cast<std::uint8_t*>(mapped), static_cast<const std::uint8_t*>(staging_src.get()) + src_offset, bytes);
    vmaUnmapMemory(m_allocator, staging_allocation);

    VkCommandBuffer command_buffer = begin_one_time_command();   // locks m_sync_mutex internally
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = dst_offset;
    copy_region.size = bytes;
    vkCmdCopyBuffer(command_buffer, staging_buffer, offload_dst.get(), 1, &copy_region);
    end_one_time_command(command_buffer);                           // locks m_sync_mutex internally

    destroy_vulkan_buffer(m_allocator, staging_buffer, staging_allocation);
}


void VulkanMemorySpace::copy_offload_to_staging(const OnHostStagingBuffer& staging_dst, size_t dst_offset, const OffloadMemoryBuffer& offload_src, size_t src_offset, size_t bytes, std::string_view site, std::string_view parameter)
{
    assert(bytes != 0);
    assert(offload_src.is_valid());
    assert(staging_dst.is_valid());

#if defined(RLLM_ENABLE_STATISTICS)
    if (!site.empty())
        parallel::statistics.record_device_to_host_buffer_copy(site, parameter, bytes);
#endif

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = VK_NULL_HANDLE;
    create_vulkan_buffer(m_allocator, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, staging_buffer, staging_allocation);

    VkCommandBuffer command_buffer = begin_one_time_command();   // locks m_sync_mutex internally
    VkBufferCopy copy_region{};
    copy_region.srcOffset = src_offset;
    copy_region.dstOffset = 0;
    copy_region.size = bytes;
    vkCmdCopyBuffer(command_buffer, offload_src.get(), staging_buffer, 1, &copy_region);
    end_one_time_command(command_buffer);                           // locks m_sync_mutex internally

    void* mapped = nullptr;
    vmaMapMemory(m_allocator, staging_allocation, &mapped);
    std::memcpy(static_cast<std::uint8_t*>(staging_dst.get()) + dst_offset, mapped, bytes);
    vmaUnmapMemory(m_allocator, staging_allocation);

    destroy_vulkan_buffer(m_allocator, staging_buffer, staging_allocation);
}


void VulkanMemorySpace::copy_offload_to_offload(const OffloadMemoryBuffer& offload_dst, size_t dst_offset, const OffloadMemoryBuffer& offload_src, size_t src_offset, size_t bytes)
{
    assert(bytes != 0);
    assert(offload_dst.is_valid());
    assert(offload_src.is_valid());

    VkCommandBuffer command_buffer = begin_one_time_command();   // locks m_sync_mutex internally
    VkBufferCopy copy_region{};
    copy_region.srcOffset = src_offset;
    copy_region.dstOffset = dst_offset;
    copy_region.size = bytes;
    vkCmdCopyBuffer(command_buffer, offload_src.get(), offload_dst.get(), 1, &copy_region);
    end_one_time_command(command_buffer);                           // locks m_sync_mutex internally
}


void VulkanMemorySpace::zero_offload(const OffloadMemoryBuffer& offload_dst, size_t offset, size_t bytes)
{
    assert(bytes != 0);
    assert(offload_dst.is_valid());

    VkCommandBuffer command_buffer = begin_one_time_command();   // locks m_sync_mutex internally
    vkCmdFillBuffer(command_buffer, offload_dst.get(), offset, bytes, 0);
    end_one_time_command(command_buffer);                           // locks m_sync_mutex internally
}


// ---- Staging (host-side) ------------------------------------------------------

OnHostStagingBuffer VulkanMemorySpace::allocate_staging(size_t bytes)
{
    assert(bytes != 0);
    void* buffer = std::malloc(bytes);
    if (buffer == nullptr)
    {
        LOG_ERROR("Failed to allocate Vulkan staging buffer of {} bytes", bytes);
        std::abort();
    }
    memset(buffer, 0, bytes);
    return OnHostStagingBuffer{buffer};
}


// ---- Offload (VMA) ------------------------------------------------------------

OffloadMemoryBuffer VulkanMemorySpace::allocate_offload(size_t bytes)
{
    // VMA-only operation — uses alloc_mutex so it does not block compute-queue
    // operations during kernel dispatch.
    std::lock_guard<std::mutex> lock(m_alloc_mutex);
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    create_vulkan_buffer(m_allocator, bytes, usage, VMA_MEMORY_USAGE_GPU_ONLY, buffer, allocation);
    return OffloadMemoryBuffer{buffer, allocation};
}


void VulkanMemorySpace::release_staging(OnHostStagingBuffer& ref)
{
    assert(ref.is_valid());
    std::free(ref.get());
    ref.invalidate();
}


void VulkanMemorySpace::release_offload(OffloadMemoryBuffer& ref)
{
    // VMA-only operation — uses alloc_mutex so it does not block compute-queue
    // operations during kernel dispatch.
    std::lock_guard<std::mutex> lock(m_alloc_mutex);
    if (ref.is_valid())
    {
        destroy_vulkan_buffer(m_allocator, ref.get(), ref.allocation());
        ref.invalidate();
    }
}


// ---- Construction / Destruction -----------------------------------------------

VulkanMemorySpace::VulkanMemorySpace()
{
    initialize_runtime();
}


// ---- One-time command helpers -------------------------------------------------
// Each helper acquires m_sync_mutex for the portion of work it performs.  This
// design means callers must NOT wrap them in an outer m_sync_mutex acquisition;
// doing so would deadlock because std::mutex is non-recursive.

VkCommandBuffer VulkanMemorySpace::begin_one_time_command()
{
    // Queue/command-pool access — protects command-buffer allocation from the
    // shared pool while also guarding against concurrent vkBeginCommandBuffer.
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    auto device = m_device;
    auto command_pool = m_command_pool;

    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandPool = command_pool;
    allocate_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkResult result = vkAllocateCommandBuffers(device, &allocate_info, &command_buffer);
    if (result != VK_SUCCESS)
    {
        LOG_ERROR("vkAllocateCommandBuffers failed: result={}", static_cast<int>(result));
        std::abort();
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (result != VK_SUCCESS)
    {
        LOG_ERROR("vkBeginCommandBuffer failed: result={}", static_cast<int>(result));
        std::abort();
    }

    return command_buffer;
}

void VulkanMemorySpace::end_one_time_command(VkCommandBuffer command_buffer)
{
    // Queue submission — protects vkQueueSubmit + vkQueueWaitIdle from concurrent
    // submissions.  Command-buffer freeing is safe once wait completes.
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    const auto device = m_device;
    const auto queue = m_queue;
    const auto command_pool = m_command_pool;

    VkResult result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS)
    {
        LOG_ERROR("vkEndCommandBuffer failed: result={}", static_cast<int>(result));
        std::abort();
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    result = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS)
    {
        LOG_ERROR("vkQueueSubmit failed: result={}", static_cast<int>(result));
        std::abort();
    }

    result = vkQueueWaitIdle(queue);
    if (result != VK_SUCCESS)
    {
        LOG_ERROR("vkQueueWaitIdle failed: result={}", static_cast<int>(result));
        std::abort();
    }

    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

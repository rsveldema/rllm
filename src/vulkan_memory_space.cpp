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
        LOG_ERROR(
            "vkEnumeratePhysicalDevices failed or found no devices: result={} physical_count={}",
            static_cast<int>(enumerate_count_result),
            physical_count
        );
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
        LOG_ERROR(
            "No Vulkan physical device with a compute-capable queue family was found. physical_count={} candidates={}",
            physical_count,
            candidates.size()
        );
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

    if (!supported_atomic_float.shaderBufferFloat32AtomicAdd
        || !supported_atomic_float.shaderSharedFloat32AtomicAdd
        || !supported_atomic_float2.shaderBufferFloat32AtomicMinMax
        || !supported_atomic_float2.shaderSharedFloat32AtomicMinMax)
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


VulkanMemorySpace::VulkanMemorySpace()
{
    initialize_runtime();
}



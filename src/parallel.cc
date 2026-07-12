#include <parallel.hpp>
#include <print>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include <vector>
#include <vulkan/vulkan.h>

IMemorySpace& IMemorySpace::get_instance()
{
    static HostMemorySpace instance;
    return instance;
}

namespace parallel {

Statistics statistics;

const char* backend_name() {
#if defined(USE_FASTFORK)
    return "fastfork";
#elif defined(USE_OPENMP)
    return "openmp";
#else
    return "sequential";
#endif
}

struct VulkanCandidate
{
    VkPhysicalDevice device = VK_NULL_HANDLE;
    uint32_t queue_family_index = 0;
    VkPhysicalDeviceProperties props{};
};

static const char* vendor_name_from_id(uint32_t vendor_id)
{
    switch (vendor_id)
    {
    case 0x10DE: return "NVIDIA";
    case 0x1002: return "AMD";
    case 0x1022: return "AMD";
    case 0x8086: return "Intel";
    case 0x13B5: return "ARM";
    case 0x5143: return "Qualcomm";
    case 0x1010: return "ImgTec";
    case 0x106B: return "Apple";
    case 0x1AE0: return "Google";
    case 0x15AD: return "VMware";
    case 0x1414: return "Microsoft";
    case 0x10005: return "Mesa";
    default: return "Unknown";
    }
}

static std::string to_lower_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static uint32_t vendor_id_from_name(const std::string& name)
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

static bool contains_case_insensitive(const std::string& haystack, const std::string& needle)
{
    if (needle.empty())
        return true;
    return to_lower_copy(haystack).find(to_lower_copy(needle)) != std::string::npos;
}

static int device_score(const VulkanCandidate& c)
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

static const VulkanCandidate* pick_candidate(const std::vector<VulkanCandidate>& candidates)
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

void print_vulkan_provider()
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

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS)
    {
        std::println("Offload provider: Vulkan (failed to create instance)");
        return;
    }

    uint32_t physical_count = 0;
    if (vkEnumeratePhysicalDevices(instance, &physical_count, nullptr) != VK_SUCCESS || physical_count == 0)
    {
        std::println("Offload provider: Vulkan (no physical device)");
        vkDestroyInstance(instance, nullptr);
        return;
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    if (vkEnumeratePhysicalDevices(instance, &physical_count, physical_devices.data()) != VK_SUCCESS)
    {
        std::println("Offload provider: Vulkan (enumeration failure)");
        vkDestroyInstance(instance, nullptr);
        return;
    }

    std::vector<VulkanCandidate> candidates;
    for (const VkPhysicalDevice pd : physical_devices)
    {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &queue_family_count, nullptr);
        if (queue_family_count == 0)
            continue;

        std::vector<VkQueueFamilyProperties> queue_props(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &queue_family_count, queue_props.data());
        for (uint32_t i = 0; i < queue_family_count; ++i)
        {
            if ((queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u)
            {
                VulkanCandidate c{};
                c.device = pd;
                c.queue_family_index = i;
                vkGetPhysicalDeviceProperties(pd, &c.props);
                candidates.push_back(c);
                break;
            }
        }
    }

    const VulkanCandidate* selected = pick_candidate(candidates);
    if (selected == nullptr)
    {
        std::println("Offload provider: Vulkan (no compute-capable device)");
        vkDestroyInstance(instance, nullptr);
        return;
    }

    std::println(
        "Offload provider: Vulkan ({}, vendor=0x{:x}, device={}, index={})",
        vendor_name_from_id(selected->props.vendorID),
        selected->props.vendorID,
        selected->props.deviceName,
        static_cast<int>(selected - candidates.data())
    );

    vkDestroyInstance(instance, nullptr);
}

} // namespace parallel

#if defined(USE_OPENMP)

namespace parallel {
    void init_parallel() {
        // Construct the active memory/offload backend before worker setup.
        (void)IMemorySpace::get_instance();
        // OMP initialises its thread pool automatically
        std::println("Using OpenMP with {} threads", get_max_threads());
    }
}

#elif defined(USE_FASTFORK)

namespace parallel {
    void init_parallel() {
        // Construct the active memory/offload backend before worker setup.
        (void)IMemorySpace::get_instance();
        fastfork::init();
        std::println("Using FastFork with {} threads", get_max_threads());
    }
}

#else // no OpenMP/FastFork: sequential execution

namespace parallel {
    void init_parallel() {
        // Construct the active memory/offload backend before worker setup.
        (void)IMemorySpace::get_instance();
        std::println("Using sequential execution");
    }
}
#endif

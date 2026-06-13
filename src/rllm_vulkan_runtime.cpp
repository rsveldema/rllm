#include <rllm_vulkan_runtime.hpp>

#if defined(USE_VULKAN_OFFLOAD)

#include <cassert>

namespace rllm::vulkan_runtime
{
    namespace
    {
        VulkanSession* g_session = nullptr;
        std::unique_ptr<VulkanComputeContext> g_context;
    }

    void set_session(VulkanSession& session)
    {
        g_session = &session;
        g_context = std::make_unique<VulkanComputeContext>(session);
    }

    VulkanSession& session()
    {
        assert(g_session != nullptr);
        return *g_session;
    }

    VulkanComputeContext& context()
    {
        assert(g_context != nullptr);
        return *g_context;
    }
}

#endif

#include <rllm_vulkan_runtime.hpp>

#if defined(USE_VULKAN_OFFLOAD)

namespace rllm::vulkan_runtime
{
    namespace
    {
        VulkanSession* g_session = nullptr;
        VulkanSession* g_owned_session = nullptr;
        VulkanComputeContext* g_context = nullptr;
        std::recursive_mutex g_mutex;

        void ensure_initialized()
        {
            std::lock_guard<std::recursive_mutex> lock(g_mutex);
            if (g_session == nullptr)
            {
                // Tests and tools may link gtest_main instead of rllm's main,
                // so they never call set_session(). Keep this process-lifetime
                // default session alive to avoid exit-time destruction ordering
                // issues with global/static model data.
                g_owned_session = new VulkanSession();
                g_session = g_owned_session;
            }
            if (g_context == nullptr)
                g_context = new VulkanComputeContext(*g_session);
        }
    }

    void set_session(VulkanSession& session)
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        g_session = &session;
        g_context = new VulkanComputeContext(session);
    }

    VulkanSession& session()
    {
        ensure_initialized();
        return *g_session;
    }

    VulkanComputeContext& context()
    {
        ensure_initialized();
        return *g_context;
    }

    std::recursive_mutex& mutex()
    {
        return g_mutex;
    }
}

#endif

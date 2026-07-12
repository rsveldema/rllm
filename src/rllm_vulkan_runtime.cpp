#include <rllm_vulkan_runtime.hpp>

namespace rllm::vulkan_runtime
{
    namespace
    {
        VulkanSession* g_session = nullptr;
        VulkanComputeContext* g_context = nullptr;
        std::recursive_mutex g_mutex;
        thread_local size_t g_queue_offset = 0;
        bool g_device_buffer_allocations_allowed = true;

    }

    void set_session(VulkanSession& session)
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        g_session = &session;
        g_context = new VulkanComputeContext(session);
    }

    VulkanSession& session()
    {
        return *g_session;
    }

    VulkanComputeContext& context()
    {
        return *g_context;
    }

    std::recursive_mutex& mutex()
    {
        return g_mutex;
    }

    VulkanQueue& get_queue(size_t index)
    {
        const size_t n = session().queue_count();
        return session().get_queue((index + g_queue_offset) % n);
    }

    size_t queue_count()
    {
        return session().queue_count();
    }

    bool device_buffer_allocations_allowed()
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        return g_device_buffer_allocations_allowed;
    }

    void set_device_buffer_allocations_allowed(bool allowed)
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        g_device_buffer_allocations_allowed = allowed;
    }

    ScopedQueueOffset::ScopedQueueOffset(size_t offset)
        : m_previous_offset(g_queue_offset)
    {
        const size_t n = queue_count();
        g_queue_offset = offset % n;
    }

    ScopedQueueOffset::~ScopedQueueOffset()
    {
        g_queue_offset = m_previous_offset;
    }
}

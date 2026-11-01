#include <rllm_hip_runtime.hpp>

namespace rllm::hip_runtime
{
    namespace
    {
        HipSession* g_session = nullptr;
        HipComputeContext* g_context = nullptr;
        std::recursive_mutex g_mutex;
        thread_local size_t g_queue_offset = 0;
        bool g_device_buffer_allocations_allowed = true;

    }

    void set_session(HipSession& session)
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        g_session = &session;
        g_context = new HipComputeContext(session);
    }

    HipSession& session()
    {
        return *g_session;
    }

    HipComputeContext& context()
    {
        return *g_context;
    }

    std::recursive_mutex& mutex()
    {
        return g_mutex;
    }

    HipQueue& get_queue(size_t index)
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

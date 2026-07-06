#include <RuntimeConfig.hpp>

#include <atomic>

namespace rllm
{
    namespace
    {
        std::atomic<bool>& nan_finding_mode_storage()
        {
            static std::atomic<bool> enabled{false};
            return enabled;
        }
    }

    bool nan_finding_mode_enabled()
    {
        return nan_finding_mode_storage().load(std::memory_order_relaxed);
    }

    void set_nan_finding_mode_enabled(bool enabled)
    {
        nan_finding_mode_storage().store(enabled, std::memory_order_relaxed);
    }
}

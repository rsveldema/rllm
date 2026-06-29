#include <RuntimeConfig.hpp>

#include <atomic>
#include <cstdlib>

namespace rllm
{
    namespace
    {
        bool env_nan_finding_mode_enabled()
        {
            const char* value = std::getenv("NAN_FINDING_MODE");
            return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
        }

        std::atomic<bool>& nan_finding_mode_storage()
        {
            static std::atomic<bool> enabled{env_nan_finding_mode_enabled()};
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

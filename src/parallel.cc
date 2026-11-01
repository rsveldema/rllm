#include <parallel.hpp>
#include <logging.hpp>
#include <print>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include <vector>
#include <hip/hip_runtime_api.h>

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

void print_hip_provider()
{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
        LOG_INFO("Offload provider: HIP (no device)");
        return;
    }
    int selected = 0;
    if (const char* value = std::getenv("RLLM_HIP_DEVICE_INDEX"))
        selected = std::clamp(static_cast<int>(std::strtol(value, nullptr, 10)), 0, count - 1);
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, selected) != hipSuccess) {
        LOG_INFO("Offload provider: HIP (device query failed)");
        return;
    }
    LOG_INFO("Offload provider: HIP ({}, index={}, memory={} MiB)", properties.name,
             selected, properties.totalGlobalMem / (1024 * 1024));
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

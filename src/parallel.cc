#include <parallel.hpp>

#if defined(USE_OPENMP)

namespace parallel {
    inline void init_parallel() {}  // OMP initialises its thread pool automatically
}

#elif defined(USE_FASTFORK)

#include <atomic>
#include <thread>

namespace parallel {
    static int s_num_threads = static_cast<int>(std::thread::hardware_concurrency());

    inline void init_parallel() {}  // fastfork initialises its thread pool automatically

    inline int get_max_threads() {
        return s_num_threads;
    }

    inline void set_num_threads(int n) {
        s_num_threads = n;
    }
}

#endif  // USE_OPENMP / USE_FASTFORK
// Sequential fallback: all functions are inline in parallel.hpp
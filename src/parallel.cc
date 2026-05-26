#include <parallel.hpp>
#include <print>

#if defined(USE_OPENMP)

namespace parallel {
    void init_parallel() {
          // OMP initialises its thread pool automatically
        std::println("Using OpenMP with {} threads", get_max_threads());
    }
}

#elif defined(USE_FASTFORK)

#include <atomic>
#include <thread>

namespace parallel {
    static int s_num_threads = static_cast<int>(std::thread::hardware_concurrency());

    void init_parallel() {
        std::println("Using FastFork with {} threads", get_max_threads());
    }

    inline int get_max_threads() {
        return s_num_threads;
    }

    inline void set_num_threads(int n) {
        s_num_threads = n;
    }
}

#else // no OpenMP/FastFork: sequential execution

namespace parallel {
    void init_parallel() {
        std::println("Using sequential execution");
    }
}
#endif 
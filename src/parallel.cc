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

namespace parallel {
    void init_parallel() {
        fastfork::init();
        std::println("Using FastFork with {} threads", get_max_threads());
    }
}

#else // no OpenMP/FastFork: sequential execution

namespace parallel {
    void init_parallel() {
        std::println("Using sequential execution");
    }
}
#endif 
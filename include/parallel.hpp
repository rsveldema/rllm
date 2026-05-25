#pragma once

#if USE_OPENMP

#include <omp.h>

namespace parallel
{
    inline int get_max_threads()
    {
        return omp_get_max_threads();
    }

    inline void set_num_threads(int num_threads)
    {
        omp_set_num_threads(num_threads);
    }
}

#else

#include "libfork/core.hpp"

namespace parallel
{
    inline int get_max_threads()
    {
        return 1;
    }

    inline void set_num_threads(int)
    {
        // no-op since we're single-threaded
    }
}

#define PARFOR(v, iterator) for (auto v : iterator) do {
#define PARFOR_2D(v1, v2, iterator) for (auto [v1, v2] : iterator) do {
#define ENDFOR } while (0)

#endif
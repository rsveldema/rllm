#pragma once

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

    inline int get_thread_num()
    {
        return 0;
    }
}

#define PARFOR(v, iterator) for (auto v : iterator) do {
#define PARFOR_2D(v1, v2, iterator) for (auto [v1, v2] : iterator) do {
#define ENDFOR } while (0)

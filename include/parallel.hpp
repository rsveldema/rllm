#pragma once

// Parallelism backend selection.
// Pass -DUSE_LIBFORK at compile time to switch to libfork.
// Default: OpenMP (requires -fopenmp).
#if !defined(USE_OPENMP) && !defined(USE_LIBFORK)
#  define USE_OPENMP
#endif

#if defined(USE_OPENMP)

#include <omp.h>

namespace parallel
{
    inline int get_max_threads() { return omp_get_max_threads(); }
    inline void set_num_threads(int n) { omp_set_num_threads(n); }
    inline int get_thread_num() { return omp_get_thread_num(); }
}

// parallel loop support:
#define PARFOR(v, ...)         _Pragma("omp parallel for schedule(static)") for (auto v : __VA_ARGS__) do {
#define PARFOR_2D(v1, v2, ...) _Pragma("omp parallel for schedule(static)") for (auto [v1, v2] : __VA_ARGS__) do {

#elif defined(USE_LIBFORK)

// TODO: libfork-based parallel loops
#include "libfork/core.hpp"

namespace parallel
{
    inline int get_max_threads() { return 1; } // TODO
    inline void set_num_threads(int) {}
    inline int get_thread_num() { return 0; }
}

#define PARFOR(v, ...)         for (auto v : __VA_ARGS__) do {
#define PARFOR_2D(v1, v2, ...) for (auto [v1, v2] : __VA_ARGS__) do {

#endif

// ENDFOR closes the do-while body opened by PARFOR / PARFOR_2D
#define ENDFOR } while (0);

// fork-join parallelism support:
#define FORK_TASK(FUNCTION_CALL) FUNCTION_CALL
#define FORK_WAIT()

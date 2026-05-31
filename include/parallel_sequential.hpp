#pragma once

#include <cstddef>

namespace parallel {
    inline int  get_max_threads()    { return 1; }
    inline void set_num_threads(int) {}
    inline int  get_thread_num()     { return 0; }
    void init_parallel();
}

#define PARFOR(v, ...)          for (auto v : (__VA_ARGS__)) {
#define PARFOR_2D(v1, v2, ...)  for (auto [v1, v2] : (__VA_ARGS__)) {
// Parallelises the lower-triangular iteration space { (v1,v2) | 0 <= v2 <= v1 < N }.
// Sequential fallback: plain nested loops.
#define PARFOR_2D_TRIANGULAR(v1, v2, N) \
    for (size_t _tri_i_ = 0; _tri_i_ < static_cast<size_t>(N); ++_tri_i_) \
        for (size_t _tri_j_ = 0; _tri_j_ <= _tri_i_; ++_tri_j_) { \
            const auto v1 = static_cast<decltype(N)>(_tri_i_); \
            const auto v2 = static_cast<decltype(N)>(_tri_j_);
// Parallelises the upper-triangular iteration space { (v1,v2) | 0 <= v1 <= v2 < N }.
// Sequential fallback: plain nested loops.
#define PARFOR_2D_UPPER_TRIANGULAR(v1, v2, N) \
    for (size_t _utri_i_ = 0, _utri_n_ = static_cast<size_t>(N); _utri_i_ < _utri_n_; ++_utri_i_) \
        for (size_t _utri_j_ = _utri_i_; _utri_j_ < _utri_n_; ++_utri_j_) { \
            const auto v1 = static_cast<decltype(N)>(_utri_i_); \
            const auto v2 = static_cast<decltype(N)>(_utri_j_);
#define ENDFOR }

#define PARSECTIONS_BEGIN  {
#define PARSECTION
#define PARSECTIONS_END    }

#define PARALLEL_DUMP_STATS() ((void)0)

// Top-level (non-nested) parallel loops that are candidates for future GPU /
// accelerator offload. Currently they expand identically to the CPU PARFOR*
// macros; the distinct name marks the intent without changing behaviour.
#define OFFLOAD_PARFOR(v, n)                     PARFOR(v, n)
#define OFFLOAD_PARFOR_2D(v1, v2, n1, n2)      PARFOR_2D(v1, v2, n1, n2)
#define OFFLOAD_PARFOR_PARAM(v, n, PARAMS) PARFOR(v, n)
#define OFFLOAD_PARFOR_2D_PARAM(v1, v2, N, PARAMS) PARFOR_2D(v1, v2, N)

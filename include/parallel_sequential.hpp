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
#define OFFLOADABLE_PARFOR(v, ...)              PARFOR(v, __VA_ARGS__)
#define OFFLOADABLE_PARFOR_2D(v1, v2, ...)      PARFOR_2D(v1, v2, __VA_ARGS__)

// Index-space OFFLOAD_PARFOR* fallback for sequential builds.
// In USE_VULKAN_OFFLOAD mode these remain CPU loops until Vulkan kernels land.
#define OFFLOAD_PARFOR(v, N) \
    for (std::size_t v = 0, _off_n_ = static_cast<std::size_t>(N); v < _off_n_; ++v) {

#define OFFLOAD_PARFOR_2D(v1, v2, N1, N2) \
    for (std::size_t v1 = 0, _off_n1_ = static_cast<std::size_t>(N1); v1 < _off_n1_; ++v1) \
        for (std::size_t v2 = 0, _off_n2_ = static_cast<std::size_t>(N2); v2 < _off_n2_; ++v2) {

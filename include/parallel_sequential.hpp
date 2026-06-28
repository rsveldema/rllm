#pragma once

#include <cstddef>

namespace parallel {
    inline int  get_max_threads()    { return 1; }
    inline void set_num_threads(int) {}
    inline int  get_thread_num()     { return 0; }
    void init_parallel();
}

#define PARFOR_1D(v, ...)          for (auto v : (__VA_ARGS__)) {
#define PARFOR_2D(v1, v2, ...)  for (auto [v1, v2] : (__VA_ARGS__)) {
#define PARFOR_3D(v1, v2, v3, ...)  for (auto [v1, v2, v3] : (__VA_ARGS__)) {
// Parallelises the lower-triangular iteration space { (v1,v2) | 0 <= v2 <= v1 < N }.
// Sequential fallback: plain nested loops.
#define PARFOR_2D_TRIANGULAR(v1, v2, N) \
    for (size_t _tri_i_ = 0; _tri_i_ < static_cast<size_t>(N); ++_tri_i_) \
        for (size_t _tri_j_ = 0; _tri_j_ <= _tri_i_; ++_tri_j_) { \
            const auto v1 = static_cast<decltype(N)>(_tri_i_); \
            const auto v2 = static_cast<decltype(N)>(_tri_j_);
#define PARFOR_3D_TRIANGULAR(v1, v2, v3, N1, N2) \
    for (size_t _tri3_h_ = 0; _tri3_h_ < static_cast<size_t>(N1); ++_tri3_h_) \
        for (size_t _tri3_i_ = 0; _tri3_i_ < static_cast<size_t>(N2); ++_tri3_i_) \
            for (size_t _tri3_j_ = 0; _tri3_j_ <= _tri3_i_; ++_tri3_j_) { \
                const auto v1 = static_cast<decltype(N1)>(_tri3_h_); \
                const auto v2 = static_cast<decltype(N2)>(_tri3_i_); \
                const auto v3 = static_cast<decltype(N2)>(_tri3_j_);
// Parallelises the upper-triangular iteration space { (v1,v2) | 0 <= v1 <= v2 < N }.
// Sequential fallback: plain nested loops.
#define PARFOR_2D_UPPER_TRIANGULAR(v1, v2, N) \
    for (size_t _utri_i_ = 0, _utri_n_ = static_cast<size_t>(N); _utri_i_ < _utri_n_; ++_utri_i_) \
        for (size_t _utri_j_ = _utri_i_; _utri_j_ < _utri_n_; ++_utri_j_) { \
            const auto v1 = static_cast<decltype(N)>(_utri_i_); \
            const auto v2 = static_cast<decltype(N)>(_utri_j_);
#define ENDFOR }


#define PARALLEL_DUMP_STATS() ((void)0)

// Top-level (non-nested) parallel loops that are candidates for future GPU /
// accelerator offload. Currently they expand identically to the CPU PARFOR*
// macros; the distinct name marks the intent without changing behaviour.
// The first argument is a VulkanQueue reference (ignored by CPU backends).
#define OFFLOAD_PARFOR_1D_PARAM(queue, v, n, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_1D(v, n)
#define OFFLOAD_PARFOR_2D_PARAM(queue, v1, v2, N, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_2D(v1, v2, N)
#define OFFLOAD_PARFOR_3D_PARAM(queue, v1, v2, v3, N, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_3D(v1, v2, v3, N)
#define OFFLOAD_PARFOR_3D_TRIANGULAR_PARAM(queue, v1, v2, v3, N1, N2, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_3D_TRIANGULAR(v1, v2, v3, N1, N2)
#define OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(queue, v1, v2, N, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_2D_TRIANGULAR(v1, v2, N)
#define OFFLOAD_PARFOR_2D_UPPER_TRIANGULAR_PARAM(queue, v1, v2, N, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_2D_UPPER_TRIANGULAR(v1, v2, N)

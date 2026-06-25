#pragma once

#include <cstddef>
#include <omp.h>

namespace parallel {
    void init_parallel();
    inline int  get_max_threads()      { return omp_get_max_threads(); }
    inline void set_num_threads(int n) { omp_set_num_threads(n); }
    inline int  get_thread_num()       { return omp_get_thread_num(); }
}

#define PARFOR_1D(v, ...)       _Pragma("omp parallel for schedule(static)") \
                                for (auto v : (__VA_ARGS__)) {
#define PARFOR_2D(v1, v2, ...)  _Pragma("omp parallel for schedule(static)") \
                                for (auto [v1, v2] : (__VA_ARGS__)) {
#define PARFOR_3D(v1, v2, v3, ...)  _Pragma("omp parallel for schedule(static)") \
                                    for (auto [v1, v2, v3] : (__VA_ARGS__)) {
// Parallelises the lower-triangular iteration space { (v1,v2) | 0 <= v2 <= v1 < N }.
// Parallelism is over the outer (v1) dimension; v2 iterates sequentially inside
// each parallel task, so accumulations into [v1,...] are race-free.
// Use schedule(dynamic) because row lengths are unequal (row v1 has v1+1 iterations).
#define PARFOR_2D_TRIANGULAR(v1, v2, N) \
    _Pragma("omp parallel for schedule(dynamic)") \
    for (long long _tri_i_ = 0; _tri_i_ < static_cast<long long>(N); ++_tri_i_) \
        for (long long _tri_j_ = 0; _tri_j_ <= _tri_i_; ++_tri_j_) { \
            const auto v1 = static_cast<decltype(N)>(_tri_i_); \
            const auto v2 = static_cast<decltype(N)>(_tri_j_);
#define PARFOR_3D_TRIANGULAR(v1, v2, v3, N1, N2) \
    _Pragma("omp parallel for schedule(dynamic)") \
    for (long long _tri3_flat_ = 0, _tri3_hn_ = static_cast<long long>(N1), _tri3_n_ = static_cast<long long>(N2); \
         _tri3_flat_ < _tri3_hn_ * _tri3_n_; \
         ++_tri3_flat_) \
        for (long long _tri3_j_ = 0; _tri3_j_ <= (_tri3_flat_ % _tri3_n_); ++_tri3_j_) { \
            const auto v1 = static_cast<decltype(N1)>(_tri3_flat_ / _tri3_n_); \
            const auto v2 = static_cast<decltype(N2)>(_tri3_flat_ % _tri3_n_); \
            const auto v3 = static_cast<decltype(N2)>(_tri3_j_);
// Parallelises the upper-triangular iteration space { (v1,v2) | 0 <= v1 <= v2 < N }.
// Parallelism is over the outer (v1) dimension; v2 iterates sequentially inside
// each parallel task, so accumulations into [v1,...] are race-free.
// Use schedule(dynamic) because row lengths are unequal (row v1 has N-v1 iterations).
#define PARFOR_2D_UPPER_TRIANGULAR(v1, v2, N) \
    _Pragma("omp parallel for schedule(dynamic)") \
    for (long long _utri_i_ = 0; _utri_i_ < static_cast<long long>(N); ++_utri_i_) \
        for (long long _utri_j_ = _utri_i_; _utri_j_ < static_cast<long long>(N); ++_utri_j_) { \
            const auto v1 = static_cast<decltype(N)>(_utri_i_); \
            const auto v2 = static_cast<decltype(N)>(_utri_j_);
#define ENDFOR }

// Parallel sections: each PARSECTION body runs concurrently.
// Usage: PARSECTIONS_BEGIN body1; PARSECTION body2; PARSECTION body3; PARSECTIONS_END
#define PARSECTIONS_BEGIN  _Pragma("omp parallel sections") { _Pragma("omp section")
#define PARSECTION         _Pragma("omp section")
#define PARSECTIONS_END    }

// No per-worker stats available under OpenMP; emit a no-op.
#define PARALLEL_DUMP_STATS() ((void)0)

// Top-level (non-nested) parallel loops that are candidates for future GPU /
// accelerator offload. Currently they expand identically to the CPU PARFOR*
// macros; the distinct name marks the intent without changing behaviour.
#define OFFLOAD_PARFOR_1D_PARAM(v, n, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_1D(v, n)
#define OFFLOAD_PARFOR_2D_PARAM(v1, v2, N, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_2D(v1, v2, N)
#define OFFLOAD_PARFOR_3D_PARAM(v1, v2, v3, N, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_3D(v1, v2, v3, N)
#define OFFLOAD_PARFOR_3D_TRIANGULAR_PARAM(v1, v2, v3, N1, N2, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_3D_TRIANGULAR(v1, v2, v3, N1, N2)
#define OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(v1, v2, N, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_2D_TRIANGULAR(v1, v2, N)
#define OFFLOAD_PARFOR_2D_UPPER_TRIANGULAR_PARAM(v1, v2, N, PARAMS) RLLM_TIMED_KERNEL(__func__) PARFOR_2D_UPPER_TRIANGULAR(v1, v2, N)

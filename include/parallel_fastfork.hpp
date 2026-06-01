#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <fastfork/fastfork.hpp>

namespace parallel {
    void init_parallel();
    inline int  get_max_threads()      { return fastfork::get_max_threads(); }
    inline void set_num_threads(int n) { fastfork::set_num_threads(n); }
    inline int  get_thread_num()       { return fastfork::get_thread_num(); }
}

// Each loop iteration is forked as an independent task.
// The extra inner scope `{ {` lets ENDFOR use `}}` to close both PARFOR
// and PARFOR_2D uniformly while still yielding one task per outer row for 2D.
#define PARFOR(v, ...) \
    { auto _ff_rng_ = (__VA_ARGS__); \
      fastfork::Context _ff_ctx_; \
      for (auto v : _ff_rng_) \
          fastfork::fork_task(_ff_ctx_, [&, v]() { {
// PARFOR_2D partitions the 2D space into dynamically-sized blocks so that
// the total task count ~= FF_PARFOR_2D_TASKS_PER_THREAD x n_threads for large
// iteration spaces, or ~= n_threads for small ones (< FF_PARFOR_2D_SMALL_THRESH
// elements total) where fork-overhead would dominate fine-grained tasks.
//
// Tile-size derivation (all at runtime so set_num_threads() is respected):
//   K       = 1 if total < threshold, else FF_PARFOR_2D_TASKS_PER_THREAD
//   tile_outer = max(1, outer_size / n_threads)
//   n_outer_blocks = ceil(outer_size / tile_outer)
//   tile_inner = bit_floor(inner_size * n_outer_blocks / (K * n_threads))
//
// Examples with K thresholds and 48 threads (large -> K=4, small -> K=1):
//     8 x 512  (small) -> tile 1 x 64  ->   8 x  8  =  64 tasks  (1.3x threads)
//    64 x 512  (small) -> tile 1 x 512 ->  64 x  1  =  64 tasks  (1.3x threads)
//    64 x 2048 (large) -> tile 1 x 512 ->  64 x  4  = 256 tasks  (5.3x threads)
//   128 x 2048 (large) -> tile 2 x 512 ->  64 x  4  = 256 tasks  (5.3x threads)
//
// Override thresholds with -DFF_PARFOR_2D_TASKS_PER_THREAD=N or
// -DFF_PARFOR_2D_SMALL_THRESH=N.
#ifndef FF_PARFOR_2D_TASKS_PER_THREAD
#  define FF_PARFOR_2D_TASKS_PER_THREAD 4
#endif
#ifndef FF_PARFOR_2D_SMALL_THRESH
#  define FF_PARFOR_2D_SMALL_THRESH 100000
#endif
#define PARFOR_2D(v1, v2, ...) \
    { auto _ff_rng_ = (__VA_ARGS__); \
      fastfork::Context _ff_ctx_; \
      const size_t _ff_os_ = _ff_rng_.outer_size(); \
      const size_t _ff_is_ = _ff_rng_.inner_size(); \
      const size_t _ff_nt_ = static_cast<size_t>(fastfork::get_max_threads()); \
      const size_t _ff_k_  = (_ff_os_ * _ff_is_ < size_t{FF_PARFOR_2D_SMALL_THRESH}) \
                             ? size_t{1} : size_t{FF_PARFOR_2D_TASKS_PER_THREAD}; \
      const size_t _ff_to_ = math::max(size_t{1}, _ff_os_ / _ff_nt_); \
      const size_t _ff_n_ob_ = (_ff_os_ + _ff_to_ - 1) / _ff_to_; \
      const size_t _ff_ti_ = std::bit_floor( \
          math::max(size_t{1}, \
                   _ff_is_ * _ff_n_ob_ / (_ff_k_ * _ff_nt_))); \
      for (size_t _ff_ob_ = 0; _ff_ob_ < _ff_os_; _ff_ob_ += _ff_to_) \
          for (size_t _ff_ib_ = 0; _ff_ib_ < _ff_is_; _ff_ib_ += _ff_ti_) \
              fastfork::fork_task(_ff_ctx_, [&, _ff_ob_, _ff_ib_, _ff_to_, _ff_ti_]() { \
                  const size_t _ff_oe_ = math::min(_ff_ob_ + _ff_to_, _ff_os_); \
                  const size_t _ff_ie_ = math::min(_ff_ib_ + _ff_ti_, _ff_is_); \
                  for (const auto [v1, v2] : _ff_rng_.block_range(_ff_ob_, _ff_oe_, _ff_ib_, _ff_ie_)) {
#define ENDFOR \
          }}); \
      }
// Minimum triangular cells each task must process. Prevents spawning so many
// tasks that O(n^2) CAS contention on the work-stealing deque top dominates.
// Tune with -DFF_PARFOR_TRIANGULAR_MIN_CELLS=N; default 256.
#ifndef FF_PARFOR_TRIANGULAR_MIN_CELLS
#  define FF_PARFOR_TRIANGULAR_MIN_CELLS 256
#endif
// Parallelises the lower-triangular iteration space { (v1,v2) | 0 <= v2 <= v1 < N }.
// Interleaved-striping: task t handles rows t, t+n_tasks, t+2*n_tasks, ... giving
// good load balance (each task gets a mix of short and long rows). The task
// count is capped at max(1, total_cells / MIN_CELLS) to avoid spawning so many
// tasks that work-stealing CAS contention exceeds the benefit of parallelism.
#define PARFOR_2D_TRIANGULAR(v1, v2, N) \
    { const size_t _tri_n_  = static_cast<size_t>(N); \
      const size_t _tri_tot_ = _tri_n_ * (_tri_n_ + 1) / 2; \
      const size_t _tri_nt_ = std::min( \
          static_cast<size_t>(fastfork::get_max_threads()), \
          std::max(size_t{1}, _tri_tot_ / size_t{FF_PARFOR_TRIANGULAR_MIN_CELLS})); \
      fastfork::Context _tri_ctx_; \
      for (size_t _tri_t_ = 0; _tri_t_ < _tri_nt_; ++_tri_t_) \
          fastfork::fork_task(_tri_ctx_, [&, _tri_t_, _tri_nt_]() { \
              for (size_t _tri_i_ = _tri_t_; _tri_i_ < _tri_n_; _tri_i_ += _tri_nt_) \
              for (size_t _tri_j_ = 0; _tri_j_ <= _tri_i_; ++_tri_j_) { \
                  const auto v1 = static_cast<decltype(N)>(_tri_i_); \
                  const auto v2 = static_cast<decltype(N)>(_tri_j_);
// Parallelises the upper-triangular iteration space { (v1,v2) | 0 <= v1 <= v2 < N }.
// Same interleaved-striping strategy and cell-count-based task cap as the lower
// triangular variant above.
#define PARFOR_2D_UPPER_TRIANGULAR(v1, v2, N) \
    { const size_t _utri_n_  = static_cast<size_t>(N); \
      const size_t _utri_tot_ = _utri_n_ * (_utri_n_ + 1) / 2; \
      const size_t _utri_nt_ = std::min( \
          static_cast<size_t>(fastfork::get_max_threads()), \
          std::max(size_t{1}, _utri_tot_ / size_t{FF_PARFOR_TRIANGULAR_MIN_CELLS})); \
      fastfork::Context _utri_ctx_; \
      for (size_t _utri_t_ = 0; _utri_t_ < _utri_nt_; ++_utri_t_) \
          fastfork::fork_task(_utri_ctx_, [&, _utri_t_, _utri_nt_]() { \
              for (size_t _utri_i_ = _utri_t_; _utri_i_ < _utri_n_; _utri_i_ += _utri_nt_) \
              for (size_t _utri_j_ = _utri_i_; _utri_j_ < _utri_n_; ++_utri_j_) { \
                  const auto v1 = static_cast<decltype(N)>(_utri_i_); \
                  const auto v2 = static_cast<decltype(N)>(_utri_j_);

#define PARSECTIONS_BEGIN  { fastfork::Context _ff_ctx_; fastfork::fork_task(_ff_ctx_, [&]() {
#define PARSECTION         }); fastfork::fork_task(_ff_ctx_, [&]() {
#define PARSECTIONS_END    }); }

// Dump per-worker fastfork statistics to stderr then reset the counters.
#define PARALLEL_DUMP_STATS() \
    do { \
        const auto _ff_stats_ = fastfork::get_worker_stats(); \
        const int  _ff_nw_    = fastfork::get_max_threads(); \
        fprintf(stderr, "  %4s  %12s  %12s  %12s  %12s\\n", \
                "tid", "exec_own", "stolen", "idle_polls", "enqueued"); \
        for (int _i_ = 0; _i_ < _ff_nw_; ++_i_) \
            fprintf(stderr, "  %4d  %12llu  %12llu  %12llu  %12llu\\n", \
                    _i_, \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_executed_own), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_stolen), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].idle_polls), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_enqueued)); \
        fastfork::reset_worker_stats(); \
    } while (false)

#if defined(USE_VULKAN_OFFLOAD) or defined(USE_HIP_OFFLOAD) or defined(USE_CUDA_OFFLOAD)
// Source offload macros are rewritten by vulkanize/hipify, but headers are not rewritten.
// Keep header-time uses valid by falling back to the CPU PARFOR forms.
#define OFFLOAD_PARFOR_1D_PARAM(v, n, PARAMS) PARFOR(v, n)
#define OFFLOAD_PARFOR_2D_PARAM(v1, v2, N, PARAMS) PARFOR_2D(v1, v2, N)
#else

// Index-space OFFLOAD_PARFOR* fallback for fastfork builds.
// With Vulkan/HIP these are located and replaced by the
// vulkanizer/hipify scripts with target-specific offload pragmas and APIs,
// but for CPU builds they just map to the regular PARFOR* macros.
#define OFFLOAD_PARFOR_1D_PARAM(v, n, PARAMS) \
    PARFOR(v, n)

#define OFFLOAD_PARFOR_2D_PARAM(v1, v2, N, PARAMS) PARFOR_2D(v1, v2, N)

#endif
#pragma once

// ── Backend selection ─────────────────────────────────────────────────────────
// Controlled via cmake: -DPARALLEL_BACKEND=openmp|fastfork|sequential
// cmake translates that into -DUSE_OPENMP or -DUSE_FASTFORK; no define → sequential.

// ── OpenMP backend ────────────────────────────────────────────────────────────
#if defined(USE_OPENMP)

#include <omp.h>

namespace parallel {
    void init_parallel();
    inline int  get_max_threads()      { return omp_get_max_threads(); }
    inline void set_num_threads(int n) { omp_set_num_threads(n); }
    inline int  get_thread_num()       { return omp_get_thread_num(); }
}

#define PARFOR(v, ...)          _Pragma("omp parallel for schedule(static)") \
                                for (auto v : (__VA_ARGS__)) {
#define PARFOR_2D(v1, v2, ...)  _Pragma("omp parallel for schedule(static)") \
                                for (auto [v1, v2] : (__VA_ARGS__)) {
#define ENDFOR }

// Parallel sections: each PARSECTION body runs concurrently.
// Usage: PARSECTIONS_BEGIN body1; PARSECTION body2; PARSECTION body3; PARSECTIONS_END
#define PARSECTIONS_BEGIN  _Pragma("omp parallel sections") { _Pragma("omp section")
#define PARSECTION         _Pragma("omp section")
#define PARSECTIONS_END    }

// No per-worker stats available under OpenMP; emit a no-op.
#define PARALLEL_DUMP_STATS() ((void)0)

// ── fastfork backend ──────────────────────────────────────────────────────────
#elif defined(USE_FASTFORK)

#include <algorithm>
#include <bit>
#include <fastfork/fastfork.hpp>

namespace parallel {
    void init_parallel();
    inline int  get_max_threads() { return fastfork::get_max_threads(); }
    inline void set_num_threads(int n) { fastfork::set_num_threads(n); }
    // fastfork does not expose per-thread IDs
    inline int get_thread_num() { return fastfork::get_thread_num(); }
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
// the total task count ≈ FF_PARFOR_2D_TASKS_PER_THREAD × n_threads for large
// iteration spaces, or ≈ n_threads for small ones (< FF_PARFOR_2D_SMALL_THRESH
// elements total) where fork-overhead would dominate fine-grained tasks.
//
// Tile-size derivation (all at runtime so set_num_threads() is respected):
//   K       = 1 if total < threshold, else FF_PARFOR_2D_TASKS_PER_THREAD
//   tile_outer = max(1, outer_size / n_threads)
//   n_outer_blocks = ceil(outer_size / tile_outer)
//   tile_inner = bit_floor(inner_size * n_outer_blocks / (K * n_threads))
//
// Examples with K thresholds and 48 threads (large → K=4, small → K=1):
//     8 × 512  (small) → tile 1 × 64  →   8 ×  8  =  64 tasks  (1.3× threads)
//    64 × 512  (small) → tile 1 × 512 →  64 ×  1  =  64 tasks  (1.3× threads)
//    64 × 2048 (large) → tile 1 × 512 →  64 ×  4  = 256 tasks  (5.3× threads)
//   128 × 2048 (large) → tile 2 × 512 →  64 ×  4  = 256 tasks  (5.3× threads)
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
      const size_t _ff_to_ = std::max(size_t{1}, _ff_os_ / _ff_nt_); \
      const size_t _ff_n_ob_ = (_ff_os_ + _ff_to_ - 1) / _ff_to_; \
      const size_t _ff_ti_ = std::bit_floor( \
          std::max(size_t{1}, \
                   _ff_is_ * _ff_n_ob_ / (_ff_k_ * _ff_nt_))); \
      for (size_t _ff_ob_ = 0; _ff_ob_ < _ff_os_; _ff_ob_ += _ff_to_) \
          for (size_t _ff_ib_ = 0; _ff_ib_ < _ff_is_; _ff_ib_ += _ff_ti_) \
              fastfork::fork_task(_ff_ctx_, [&, _ff_ob_, _ff_ib_, _ff_to_, _ff_ti_]() { \
                  const size_t _ff_oe_ = std::min(_ff_ob_ + _ff_to_, _ff_os_); \
                  const size_t _ff_ie_ = std::min(_ff_ib_ + _ff_ti_, _ff_is_); \
                  for (const auto [v1, v2] : _ff_rng_.block_range(_ff_ob_, _ff_oe_, _ff_ib_, _ff_ie_)) {
#define ENDFOR \
          }}); \
      }

#define PARSECTIONS_BEGIN  { fastfork::Context _ff_ctx_; fastfork::fork_task(_ff_ctx_, [&]() {
#define PARSECTION         }); fastfork::fork_task(_ff_ctx_, [&]() {
#define PARSECTIONS_END    }); }

// Dump per-worker fastfork statistics to stderr then reset the counters.
#define PARALLEL_DUMP_STATS() \
    do { \
        const auto _ff_stats_ = fastfork::get_worker_stats(); \
        const int  _ff_nw_    = fastfork::get_max_threads(); \
        fprintf(stderr, "  %4s  %12s  %12s  %12s  %12s\n", \
                "tid", "exec_own", "stolen", "idle_polls", "enqueued"); \
        for (int _i_ = 0; _i_ < _ff_nw_; ++_i_) \
            fprintf(stderr, "  %4d  %12llu  %12llu  %12llu  %12llu\n", \
                    _i_, \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_executed_own), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_stolen), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].idle_polls), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_enqueued)); \
        fastfork::reset_worker_stats(); \
    } while (false)

// ── Sequential fallback ───────────────────────────────────────────────────────
#else

namespace parallel {
    inline int  get_max_threads()      { return 1; }
    inline void set_num_threads(int)   {}
    inline int  get_thread_num()       { return 0; }
    void init_parallel();
}

#define PARFOR(v, ...)          for (auto v : (__VA_ARGS__)) {
#define PARFOR_2D(v1, v2, ...)  for (auto [v1, v2] : (__VA_ARGS__)) {
#define ENDFOR }

#define PARSECTIONS_BEGIN  {
#define PARSECTION
#define PARSECTIONS_END    }

#define PARALLEL_DUMP_STATS() ((void)0)

#endif

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

// ── fastfork backend ──────────────────────────────────────────────────────────
#elif defined(USE_FASTFORK)

#include <fastfork/fastfork.hpp>

namespace parallel {
    inline void init_parallel() { fastfork::init(); }
    inline int  get_max_threads() { return fastfork::get_max_threads(); }
    inline void set_num_threads(int n) { fastfork::set_num_threads(n); }
    // fastfork does not expose per-thread IDs
    inline int get_thread_num() { return fastfork::get_thread_num(); }
}

// Each loop iteration is forked as an independent task; ENDFOR waits for all.
// Loop variables are captured by value so each task sees its own copy.
#define PARFOR(v, ...) \
    { auto _ff_rng_ = (__VA_ARGS__); \
      for (auto v : _ff_rng_) \
          fastfork::fork_task([&, v]() {
#define PARFOR_2D(v1, v2, ...) \
    { auto _ff_rng_ = (__VA_ARGS__); \
      for (auto [v1, v2] : _ff_rng_) \
          fastfork::fork_task([&, v1 = v1, v2 = v2]() {
#define ENDFOR \
      }); \
      fastfork::wait_for_all_tasks(); }

#define PARSECTIONS_BEGIN  { fastfork::fork_task([&]() {
#define PARSECTION         }); fastfork::fork_task([&]() {
#define PARSECTIONS_END    }); fastfork::wait_for_all_tasks(); }

// ── Sequential fallback ───────────────────────────────────────────────────────
#else

namespace parallel {
    inline int  get_max_threads()      { return 1; }
    inline void set_num_threads(int)   {}
    inline int  get_thread_num()       { return 0; }
    inline void init_parallel()        {}
}

#define PARFOR(v, ...)          for (auto v : (__VA_ARGS__)) {
#define PARFOR_2D(v1, v2, ...)  for (auto [v1, v2] : (__VA_ARGS__)) {
#define ENDFOR }

#define PARSECTIONS_BEGIN  {
#define PARSECTION
#define PARSECTIONS_END    }

#endif

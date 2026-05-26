#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace fastfork
{
    using task_t = std::function<void()>;

    void init();

    // Returns the maximum number of threads available for parallel execution.
    int get_max_threads();

    // how many threads to use for parallel regions; default is number of hardware threads
    void set_num_threads(int n);

    // returns thread i of N
    int get_thread_num();


    // Fork a task to run in parallel. The task will be executed by a worker thread.
    // it is allowed for a task to fork additional tasks, which will also be executed in parallel.
    // We are NUMA aware, so tasks may be scheduled on different NUMA nodes, and we will attempt
    // to schedule tasks on the same NUMA node as the data they access.
    void fork_task(task_t t);

    // Scoped batch handle.
    //   fork_task(ctx, t)  — increments ctx, wraps t to decrement on completion.
    //   ctx destructor     — calls wait_local(ctx) so the batch always drains
    //                        before ctx goes out of scope.
    // Outer in-flight tasks are not counted, so nested fork/wait is deadlock-free.
    class Context
    {
    public:
        Context() = default;
        ~Context();                         // defined in fastfork.cc; calls wait_local
        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

        void operator++() noexcept { m_n.fetch_add(1, std::memory_order_relaxed); }
        void operator--() noexcept { m_n.fetch_sub(1, std::memory_order_release); }
        bool empty()  const noexcept { return m_n.load(std::memory_order_acquire) == 0; }

    private:
        std::atomic<int> m_n{0};
    };

    // Fork a task and register it with a Context batch.
    void fork_task(Context& ctx, task_t t);

    // Participate in work-stealing until ctx is empty.
    void wait_local(Context& ctx);

    // Per-worker statistics (indexed by thread ID 0..get_max_threads()-1).
    struct WorkerStats
    {
        uint64_t tasks_executed_own{0}; // tasks run from thread's own queue
        uint64_t tasks_stolen{0};       // tasks run stolen from another thread
        uint64_t idle_polls{0};         // full steal_order scans that found nothing
        uint64_t tasks_enqueued{0};     // tasks deposited into this thread's queue
    };

    std::vector<WorkerStats> get_worker_stats();
    void                     reset_worker_stats();
}
#pragma once

#include <functional>

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

    // Wait for all forked tasks to complete.
    void wait_for_all_tasks();
}
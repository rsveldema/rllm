#include <fastfork/fastfork.hpp>

#include <hwloc.h>

namespace fastfork
{
    // workstealing task scheduler implementation goes here.
    // You can use a thread pool and a concurrent queue of tasks.
    // we are NUMA aware using hwloc

    void init()
    {
    }

        // Returns the maximum number of threads available for parallel execution.
    int get_max_threads();

    // how many threads to use for parallel regions; default is number of hardware threads
    void set_num_threads(int n);

    // returns thread i of N
    int get_thread_num();


    void fork_task(task_t t)
    {
        // Implementation of task scheduling goes here
    }

    void wait_for_all_tasks()
    {
        // Implementation of waiting for all tasks to complete goes here
    }
}
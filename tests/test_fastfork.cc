#include <gtest/gtest.h>

#include <fastfork/fastfork.hpp>

#include <atomic>
#include <chrono>
#include <print>
#include <thread>
#include <vector>

// fastfork::init() is called once in main() below.

// ── helpers ────────────────────────────────────────────────────────────────

static void fork_n_tasks(int n, std::atomic<int>& counter)
{
    fastfork::Context ctx;
    for (int i = 0; i < n; ++i)
        fastfork::fork_task(ctx, [&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
}

// ── tests ──────────────────────────────────────────────────────────────────

TEST(FastforkTest, GetMaxThreadsPositive)
{
    EXPECT_GT(fastfork::get_max_threads(), 0);
}

// Empty Context destruction with nothing queued must return immediately.
TEST(FastforkTest, WaitWithNoTasks)
{
    const auto t0 = std::chrono::steady_clock::now();
    { fastfork::Context ctx; }
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 100);
}

TEST(FastforkTest, SingleTask)
{
    std::atomic<int> ran{0};
    {
        fastfork::Context ctx;
        fastfork::fork_task(ctx, [&ran] { ran.store(1, std::memory_order_release); });
    }
    EXPECT_EQ(ran.load(), 1);
}

TEST(FastforkTest, AllTasksRun)
{
    constexpr int N = 1000;
    std::atomic<int> counter{0};
    fork_n_tasks(N, counter);
    EXPECT_EQ(counter.load(), N);
}

// Multiple fork+wait cycles must each complete correctly.
TEST(FastforkTest, RepeatedCycles)
{
    constexpr int CYCLES = 10;
    constexpr int N      = 200;
    for (int c = 0; c < CYCLES; ++c)
    {
        std::atomic<int> counter{0};
        fork_n_tasks(N, counter);
        EXPECT_EQ(counter.load(), N) << "cycle " << c;
    }
}

// A task is allowed to fork sub-tasks.
TEST(FastforkTest, NestedFork)
{
    constexpr int OUTER = 8;
    constexpr int INNER = 16;
    std::atomic<int> counter{0};
    {
        fastfork::Context outer_ctx;
        for (int i = 0; i < OUTER; ++i)
        {
            fastfork::fork_task(outer_ctx, [&counter]
            {
                fastfork::Context inner_ctx;
                for (int j = 0; j < INNER; ++j)
                    fastfork::fork_task(inner_ctx, [&counter]
                    {
                        counter.fetch_add(1, std::memory_order_relaxed);
                    });
            });
        }
    }
    EXPECT_EQ(counter.load(), OUTER * INNER);
}

// get_thread_num() must be in [0, max_threads) for every executing task.
TEST(FastforkTest, ThreadIdInRange)
{
    const int max = fastfork::get_max_threads();
    std::atomic<bool> bad{false};
    constexpr int N = 500;
    {
        fastfork::Context ctx;
        for (int i = 0; i < N; ++i)
        {
            fastfork::fork_task(ctx, [max, &bad]
            {
                const int id = fastfork::get_thread_num();
                if (id < 0 || id >= max)
                    bad.store(true, std::memory_order_relaxed);
            });
        }
    }
    EXPECT_FALSE(bad.load());
}

// Verify tasks spread across more than one thread (only meaningful with >1 thread).
TEST(FastforkTest, TasksRunOnMultipleThreads)
{
    const int max = fastfork::get_max_threads();
    if (max < 2) GTEST_SKIP() << "only one thread available";

    std::vector<std::atomic<int>> seen(max);
    for (auto& a : seen) a.store(0, std::memory_order_relaxed);

    constexpr int N = 2000;
    {
        fastfork::Context ctx;
        for (int i = 0; i < N; ++i)
        {
            fastfork::fork_task(ctx, [&seen]
            {
                seen[fastfork::get_thread_num()].fetch_add(1, std::memory_order_relaxed);
            });
        }
    }

    int active = 0;
    for (auto& a : seen) if (a.load() > 0) ++active;
    EXPECT_GT(active, 1) << "all tasks ran on a single thread";
}

// set_num_threads() restarts the pool; tasks must still complete correctly.
TEST(FastforkTest, SetNumThreadsRespawns)
{
    const int original = fastfork::get_max_threads();
    const int reduced  = std::max(1, original / 2);

    fastfork::set_num_threads(reduced);
    EXPECT_EQ(fastfork::get_max_threads(), reduced);

    constexpr int N = 500;
    std::atomic<int> counter{0};
    fork_n_tasks(N, counter);
    EXPECT_EQ(counter.load(), N);

    // restore
    fastfork::set_num_threads(original);
    EXPECT_EQ(fastfork::get_max_threads(), original);
    counter.store(0);
    fork_n_tasks(N, counter);
    EXPECT_EQ(counter.load(), N);
}

// ── speedup ──────────────────────────────────────────────────────────────

static double slow_approximate_sqrt(double x)
{
    double r = x * 0.5;
    for (int k = 0; k < 8; ++k) r = 0.5 * (r + x / r);
    return r;
}

// Run an embarrassingly parallel CPU-bound loop serially then via fastfork
// and log the observed speedup.  Asserts speedup > 1 on multi-core machines.
TEST(FastforkTest, SpeedupEmbarrassinglyParallel)
{
    const int max = fastfork::get_max_threads();
    if (max < 2) GTEST_SKIP() << "only one thread available";

    // 512 independent chunks, each approximating sqrt via Newton's method
    constexpr int CHUNKS = 512 * 32;
    constexpr int WORK   = 20'000;
    std::vector<double> results(CHUNKS);

    // serial
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < CHUNKS; ++c)
    {
        double sum = 0.0;
        for (int j = 0; j < WORK; ++j)
            sum += slow_approximate_sqrt(static_cast<double>(c * WORK + j + 1));
        results[c] = sum;
    }
    const auto t1 = std::chrono::steady_clock::now();

    // parallel
    std::fill(results.begin(), results.end(), 0.0);
    const auto t2 = std::chrono::steady_clock::now();
    {
        fastfork::Context ctx;
        for (int c = 0; c < CHUNKS; ++c)
            fastfork::fork_task(ctx, [c, &results]
            {
                double sum = 0.0;
                for (int j = 0; j < WORK; ++j)
                    sum += slow_approximate_sqrt(static_cast<double>(c * WORK + j + 1));
                results[c] = sum;
            });
    }
    const auto t3 = std::chrono::steady_clock::now();

    const long long serial_us   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const long long parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double    speedup     = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us",   serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup_x",   speedup);
    std::println("FastFork speedup ({} threads) - Serial: {}us, Parallel: {}us, Speedup: {:.2f}x",
                 max, serial_us, parallel_us, speedup);

    // Print per-worker stats to reveal load balance and steal behaviour.
    const auto stats = fastfork::get_worker_stats();
    std::println("  {:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
                 "tid", "exec_own", "stolen", "idle_polls", "enqueued");
    for (int i = 0; i < max; ++i)
        std::println("  {:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
                     i,
                     stats[i].tasks_executed_own,
                     stats[i].tasks_stolen,
                     stats[i].idle_polls,
                     stats[i].tasks_enqueued);

    EXPECT_GT(speedup, 1.0) << "parallel was not faster than serial";
}

// ── overhead check ────────────────────────────────────────────────────────

// Dispatch N trivially-cheap tasks through fastfork and verify the
// per-task scheduling overhead stays below a reasonable ceiling.
// This catches regressions in the critical path (atomic ops, mutex, etc.).
TEST(FastforkTest, OverheadPerTaskBounded)
{
    constexpr int       N                     = 10'000 * 100; // 1 million tasks
    constexpr long long MAX_OVERHEAD_US_PER_TASK = 50; // 50 µs per task

    std::atomic<int> counter{0};

    // serial baseline: N direct atomic increments (fastest possible work)
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
        counter.fetch_add(1, std::memory_order_relaxed);
    const auto t1 = std::chrono::steady_clock::now();
    ASSERT_EQ(counter.load(), N);
    counter.store(0);

    // fastfork: same work dispatched as N individual tasks
    const auto t2 = std::chrono::steady_clock::now();
    {
        fastfork::Context ctx;
        for (int i = 0; i < N; ++i)
            fastfork::fork_task(ctx, [&counter]
            {
                counter.fetch_add(1, std::memory_order_relaxed);
            });
    }
    const auto t3 = std::chrono::steady_clock::now();
    ASSERT_EQ(counter.load(), N);

    const long long serial_us   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const long long parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const long long overhead_us = parallel_us - serial_us;
    const long long overhead_per_task_us = overhead_us / N;

    RecordProperty("serial_us",            serial_us);
    RecordProperty("parallel_us",          parallel_us);
    RecordProperty("overhead_per_task_us", overhead_per_task_us);
    std::println("FastFork overhead - Serial: {}us, Parallel: {}us, Overhead/task: {}us",
                 serial_us, parallel_us, overhead_per_task_us);

    EXPECT_LT(overhead_per_task_us, MAX_OVERHEAD_US_PER_TASK)
        << "per-task fastfork overhead exceeded " << MAX_OVERHEAD_US_PER_TASK << " µs";
}

// ── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    fastfork::init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

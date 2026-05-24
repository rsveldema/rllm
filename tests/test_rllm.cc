#include <gtest/gtest.h>

#include <TransformerBlock.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>

#include <omp.h>

TEST(PredictorTest, Placeholder) {
    SUCCEED();
}

namespace
{
    constexpr int BENCH_ITERS          = 20;
    constexpr int TEST_SEQ_LEN         = 8;   // smoke tests
    constexpr int BENCH_SEQ_LEN        = 64;  // speedup benchmark needs more work
} // namespace

// ---------------------------------------------------------------------------
// TransformerBlock smoke test: forward produces output of correct shape
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, ForwardOutputShape)
{
    auto block = std::make_unique<rllm::TransformerBlock>();
    block->randomize();

    const int T = TEST_SEQ_LEN;
    const int D = rllm::TransformerBlock::D_MODEL;
    std::vector<float> h(T * D, 0.1f);

    block->forward(h, T);

    ASSERT_EQ(static_cast<int>(h.size()), T * D)
        << "forward() must not change the size of h";
}

// ---------------------------------------------------------------------------
// TransformerBlock smoke test: backward runs and returns correct shape
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, BackwardOutputShape)
{
    auto block = std::make_unique<rllm::TransformerBlock>();
    block->randomize();

    const int T = TEST_SEQ_LEN;
    const int D = rllm::TransformerBlock::D_MODEL;
    std::vector<float> h(T * D, 0.05f);
    block->forward(h, T);

    std::vector<float> dout(T * D, 0.01f);
    const auto din = block->backward(dout, 0.01f);

    ASSERT_EQ(static_cast<int>(din.size()), T * D)
        << "backward() must return a gradient of the same size as the input";
}

// ---------------------------------------------------------------------------
// TransformerBlock OpenMP test: parallel forward faster than serial
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, ForwardParallelFasterThanSerial)
{
    if (omp_get_max_threads() < 2)
        GTEST_SKIP() << "OpenMP thread count < 2 - no parallelism available";

    const int max_threads = omp_get_max_threads();

    auto block = std::make_unique<rllm::TransformerBlock>();
    block->randomize();

    const int T = BENCH_SEQ_LEN;
    const int D = rllm::TransformerBlock::D_MODEL;
    const std::vector<float> h_template(T * D, 0.1f);

    // --- serial baseline (1 thread) ---
    omp_set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        std::vector<float> h = h_template;
        block->forward(h, T);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- OpenMP parallel ---
    omp_set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        std::vector<float> h = h_template;
        block->forward(h, T);
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup   = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us",   serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup",     speedup);

    fprintf(stderr, "TransformerBlock Forward - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
            static_cast<long long>(serial_us), static_cast<long long>(parallel_us), speedup);

    EXPECT_GT(speedup, 1.0)
        << "OpenMP parallelisation of TransformerBlock::forward is slower than serial "
        << "(serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup << ").";
}

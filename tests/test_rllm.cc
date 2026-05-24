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
    const int D = static_cast<int>(rllm::EmbeddingDimension::MAX);
    rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> h(
        static_cast<rllm::PositionIndex>(T));
    h.fill(0.1f);

    block->forward(h, static_cast<rllm::PositionIndex>(T));

    ASSERT_EQ(static_cast<size_t>(h.num_rows()), static_cast<size_t>(T));
}

// ---------------------------------------------------------------------------
// TransformerBlock smoke test: backward runs and returns correct shape
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, BackwardOutputShape)
{
    auto block = std::make_unique<rllm::TransformerBlock>();
    block->randomize();

    const int T = TEST_SEQ_LEN;
    const int D = static_cast<int>(rllm::EmbeddingDimension::MAX);
    rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> h(
        static_cast<rllm::PositionIndex>(T));
    h.fill(0.05f);
    block->forward(h, static_cast<rllm::PositionIndex>(T));

    rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> dout(
        static_cast<rllm::PositionIndex>(T));
    dout.fill(0.01f);
    rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
        static_cast<rllm::PositionIndex>(T));
    block->backward(dout, din, 0.01f);

    ASSERT_EQ(static_cast<int>(din.num_rows()) * static_cast<int>(din.num_cols()), T * D)
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
    const int D = static_cast<int>(rllm::EmbeddingDimension::MAX);
    rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> h_template(
        static_cast<rllm::PositionIndex>(T));
    h_template.fill(0.1f);

    // --- serial baseline (1 thread) ---
    omp_set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        auto h = h_template;
        block->forward(h, static_cast<rllm::PositionIndex>(T));
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- OpenMP parallel ---
    omp_set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        auto h = h_template;
        block->forward(h, static_cast<rllm::PositionIndex>(T));
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

// ---------------------------------------------------------------------------
// TransformerBlock OpenMP test: parallel backward faster than serial
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, BackwardParallelFasterThanSerial)
{
    if (omp_get_max_threads() < 2)
        GTEST_SKIP() << "OpenMP thread count < 2 - no parallelism available";

    const int max_threads = omp_get_max_threads();

    auto block = std::make_unique<rllm::TransformerBlock>();
    block->randomize();

    const int T = BENCH_SEQ_LEN;
    rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> h_template(
        static_cast<rllm::PositionIndex>(T));
    h_template.fill(0.1f);
    rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> dout_template(
        static_cast<rllm::PositionIndex>(T));
    dout_template.fill(0.01f);

    // Prime the block with a forward pass so backward has valid cached state.
    {
        auto h = h_template;
        block->forward(h, static_cast<rllm::PositionIndex>(T));
    }

    // --- serial baseline (1 thread) ---
    omp_set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
            static_cast<rllm::PositionIndex>(T));
        block->backward(dout_template, din, 0.01f);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- OpenMP parallel ---
    omp_set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
            static_cast<rllm::PositionIndex>(T));
        block->backward(dout_template, din, 0.01f);
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup   = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us",   serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup",     speedup);

    fprintf(stderr, "TransformerBlock Backward - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
            static_cast<long long>(serial_us), static_cast<long long>(parallel_us), speedup);

    EXPECT_GT(speedup, 1.0)
        << "OpenMP parallelisation of TransformerBlock::backward is slower than serial "
        << "(serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup << ").";
}

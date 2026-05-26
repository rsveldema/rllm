#include <gtest/gtest.h>

#include <TransformerBlock.hpp>
#include <parallel.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>

int main(int argc, char** argv)
{
    parallel::init_parallel();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(PredictorTest, Placeholder) {
    SUCCEED();
}

namespace
{
    constexpr int BENCH_ITERS          = 20;
    constexpr int BENCH_FORWARD_ITERS  = 200; // forward is fast; needs more iters for stable timing
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
// TransformerBlock test: parallel forward faster than serial
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, ForwardParallelFasterThanSerial)
{
    if (parallel::get_max_threads() < 2)
        GTEST_SKIP() << "thread count < 2 - no parallelism available";

    const int max_threads = parallel::get_max_threads();

    auto block = std::make_unique<rllm::TransformerBlock>();
    block->randomize();

    const int T = BENCH_SEQ_LEN;
    const int D = static_cast<int>(rllm::EmbeddingDimension::MAX);
    rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> h_template(
        static_cast<rllm::PositionIndex>(T));
    h_template.fill(0.1f);

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_FORWARD_ITERS; ++iter)
    {
        auto h = h_template;
        block->forward(h, static_cast<rllm::PositionIndex>(T));
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel ---
    parallel::set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_FORWARD_ITERS; ++iter)
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

    fprintf(stdout, "TransformerBlock Forward - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
            static_cast<long long>(serial_us), static_cast<long long>(parallel_us), speedup);

    EXPECT_GT(speedup, 1.0)
        << "parallelisation of TransformerBlock::forward is slower than serial "
        << "(serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup << ")."
;
    PARALLEL_DUMP_STATS();
}

// ---------------------------------------------------------------------------
// TransformerBlock test: parallel backward faster than serial
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, BackwardParallelFasterThanSerial)
{
    if (parallel::get_max_threads() < 2)
        GTEST_SKIP() << "thread count < 2 - no parallelism available";

    const int max_threads = parallel::get_max_threads();

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
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        rllm::flexible_rows_matrix<float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
            static_cast<rllm::PositionIndex>(T));
        block->backward(dout_template, din, 0.01f);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel ---
    parallel::set_num_threads(max_threads);
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

    fprintf(stdout, "TransformerBlock Backward - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
            static_cast<long long>(serial_us), static_cast<long long>(parallel_us), speedup);
    PARALLEL_DUMP_STATS();

    EXPECT_GT(speedup, 1.0)
        << "parallelisation of TransformerBlock::backward is slower than serial "
        << "(serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup << ").";
}

// ---------------------------------------------------------------------------
// enum_iterator2D tests
// ---------------------------------------------------------------------------

// Visits every (Enum1, Enum2) pair exactly once (full 8×64 = 512 pairs).
TEST(EnumIterator2DTest, TotalCount)
{
    using namespace rllm;
    int count = 0;
    for (const auto [h, d] : enum_iterator2D<HeadsIndex, HeadDimension>())
        ++count;
    constexpr int expected = static_cast<int>(HeadsIndex::MAX) * static_cast<int>(HeadDimension::MAX);
    EXPECT_EQ(count, expected);
}

// First enum (Enum1) is the outer/slow dimension; second (Enum2) is the inner/fast dimension.
TEST(EnumIterator2DTest, RowMajorOrder)
{
    using namespace rllm;
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<int>(HeadsIndex::MAX) * static_cast<int>(HeadDimension::MAX));
    for (const auto [h, d] : enum_iterator2D<HeadsIndex, HeadDimension>())
        pairs.emplace_back(static_cast<int>(h), static_cast<int>(d));

    int idx = 0;
    for (int h = 0; h < static_cast<int>(HeadsIndex::MAX); ++h)
        for (int d = 0; d < static_cast<int>(HeadDimension::MAX); ++d, ++idx)
        {
            EXPECT_EQ(pairs[idx].first,  h) << "wrong outer index at flat index " << idx;
            EXPECT_EQ(pairs[idx].second, d) << "wrong inner index at flat index " << idx;
        }
}

// Runtime-bounded first dimension: end1 < Enum1::MAX.
TEST(EnumIterator2DTest, RuntimeBound)
{
    using namespace rllm;
    constexpr int bound = 3; // only first 3 heads
    int count = 0;
    for (const auto [h, d] : enum_iterator2D<HeadsIndex, HeadDimension>(static_cast<HeadsIndex>(bound)))
        ++count;
    EXPECT_EQ(count, bound * static_cast<int>(HeadDimension::MAX));
}

// A zero bound on the first dimension yields no iterations.
TEST(EnumIterator2DTest, ZeroBoundProducesNoIterations)
{
    using namespace rllm;
    int count = 0;
    for (const auto [h, d] : enum_iterator2D<HeadsIndex, HeadDimension>(HeadsIndex::START))
        ++count;
    EXPECT_EQ(count, 0);
}

// ---------------------------------------------------------------------------
// PARFOR_2D performance test: parallel 2D loop faster than serial
// ---------------------------------------------------------------------------
// Dimensions mirror the most common PARFOR_2D usage in TransformerBlock:
// PositionIndex (outer, BENCH_SEQ_LEN rows) x EmbeddingDimension (inner, 512 cols).
// Each cell does WORK_PER_CELL float multiply-adds so that fork overhead is
// clearly amortised even if task count is small.
TEST(ParFor2DTest, SpeedupFasterThanSerial)
{
    if (parallel::get_max_threads() < 2)
        GTEST_SKIP() << "thread count < 2 - no parallelism available";

    const int max_threads = parallel::get_max_threads();

    constexpr int ROWS          = BENCH_SEQ_LEN;                                   // 64
    constexpr int COLS          = static_cast<int>(rllm::EmbeddingDimension::MAX); // 512
    constexpr int WORK_PER_CELL = 128; // float multiply-adds per cell
    constexpr int ITERS         = 100;

    const auto seq = static_cast<rllm::PositionIndex>(ROWS);
    std::vector<float> buf(ROWS * COLS, 0.0f);

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it)
    {
        for (const auto [t, d] :
             rllm::enum_iterator2D<rllm::PositionIndex, rllm::EmbeddingDimension>(seq))
        {
            float v = static_cast<float>(static_cast<int>(t) * COLS + static_cast<int>(d) + it);
            for (int k = 0; k < WORK_PER_CELL; ++k)
                v = v * 1.00001f + 0.00001f;
            buf[static_cast<int>(t) * COLS + static_cast<int>(d)] = v;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel (PARFOR_2D) ---
    parallel::set_num_threads(max_threads);
    std::fill(buf.begin(), buf.end(), 0.0f);
    const auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it)
    {
        PARFOR_2D(t, d, rllm::enum_iterator2D<rllm::PositionIndex, rllm::EmbeddingDimension>(seq))
            float v = static_cast<float>(static_cast<int>(t) * COLS + static_cast<int>(d) + it);
            for (int k = 0; k < WORK_PER_CELL; ++k)
                v = v * 1.00001f + 0.00001f;
            buf[static_cast<int>(t) * COLS + static_cast<int>(d)] = v;
        ENDFOR
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup   = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us",   serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup",     speedup);

    fprintf(stderr,
            "PARFOR_2D (%d outer x %d inner, work=%d, iters=%d, threads=%d)"
            " - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
            ROWS, COLS, WORK_PER_CELL, ITERS, max_threads,
            static_cast<long long>(serial_us),
            static_cast<long long>(parallel_us),
            speedup);

    EXPECT_GT(speedup, 1.0)
        << "PARFOR_2D was not faster than serial"
        << " (serial=" << serial_us << "us, parallel=" << parallel_us
        << "us, speedup=" << speedup << ").";
}

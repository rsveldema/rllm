#include <gtest/gtest.h>

#include <Corpus.hpp>
#include <NeuralNetwork.hpp>
#include <Statistics.hpp>
#include <TransformerBlock.hpp>
#include <parallel.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <print>
#include <vector>

using rllm::rlmm_float;

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

    std::unique_ptr<rllm::NeuralNetwork> load_guaranteed_model_or_skip(
        rllm::Corpus& corpus,
        rllm::Statistics& stats
    )
    {
        const char* model_path = "models/guaranteed_fresh.json";
        if (!std::filesystem::exists(model_path))
            return nullptr;

        auto nn = std::make_unique<rllm::NeuralNetwork>(4, corpus, stats);
        if (!nn->load(model_path))
            return nullptr;

        return nn;
    }

    std::vector<rllm::OutputToken> top5_for_prompt(
        rllm::NeuralNetwork& nn,
        rllm::Corpus& corpus,
        const std::string& prompt
    )
    {
        const auto token_ids = corpus.get_token_ids(prompt);
        nn.propagate_forward(token_ids);
        const auto top5 = nn.get_best_output_token_ids(5);

        std::println("Prompt '{}', top-5:", prompt);
        for (size_t i = 0; i < top5.size(); ++i)
        {
            const auto tok = corpus.get_token_from_id(top5[i].token_id);
            std::println(
                "  [{}] token='{}' id={} p={:.6f}",
                i,
                tok,
                static_cast<int>(top5[i].token_id),
                top5[i].activation
            );
        }
        return top5;
    }
} // namespace

TEST(PredictorRegressionTest, GuaranteedModel_HashPredictsInclude)
{
    std::vector<std::string> filters = {"guaranteed"};
    rllm::Corpus corpus(filters);
    rllm::Statistics stats;
    auto nn = load_guaranteed_model_or_skip(corpus, stats);
    if (!nn)
        GTEST_SKIP() << "Missing or unloadable model fixture: models/guaranteed_fresh.json";

    const auto top5 = top5_for_prompt(*nn, corpus, "#");
    ASSERT_FALSE(top5.empty());
    EXPECT_EQ(corpus.get_token_from_id(top5.front().token_id), "include");
}

TEST(PredictorRegressionTest, GuaranteedModel_IncludePredictsA)
{
    std::vector<std::string> filters = {"guaranteed"};
    rllm::Corpus corpus(filters);
    rllm::Statistics stats;
    auto nn = load_guaranteed_model_or_skip(corpus, stats);
    if (!nn)
        GTEST_SKIP() << "Missing or unloadable model fixture: models/guaranteed_fresh.json";

    const auto top5 = top5_for_prompt(*nn, corpus, "#include");
    ASSERT_FALSE(top5.empty());
    EXPECT_EQ(corpus.get_token_from_id(top5.front().token_id), "A");
}

// ---------------------------------------------------------------------------
// TransformerBlock smoke test: forward produces output of correct shape
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, ForwardOutputShape)
{
    auto block = std::make_unique<rllm::TransformerBlock>();
    block->randomize();

    const int T = TEST_SEQ_LEN;
    const int D = static_cast<int>(rllm::EmbeddingDimension::MAX);
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> h(
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
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> h(
        static_cast<rllm::PositionIndex>(T));
    h.fill(0.05f);
    block->forward(h, static_cast<rllm::PositionIndex>(T));

    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> dout(
        static_cast<rllm::PositionIndex>(T));
    dout.fill(0.01f);
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
        static_cast<rllm::PositionIndex>(T));
    block->backward(dout, din, 0.01f);

    ASSERT_EQ(static_cast<int>(din.num_rows()) * static_cast<int>(din.num_cols()), T * D)
        << "backward() must return a gradient of the same size as the input";
}

TEST(TransformerBlockTest, CausalSoftmaxMasksFutureTokensAndNormalizesRows)
{
    using namespace rllm;

    const auto T = static_cast<PositionIndex>(4);
    flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex> scores(T, T);

    // Distinct values per row; future positions (j > i) should be ignored then zeroed.
    scores[static_cast<PositionIndex>(0), static_cast<PositionIndex>(0)] = 1.0f;
    scores[static_cast<PositionIndex>(0), static_cast<PositionIndex>(1)] = 999.0f;
    scores[static_cast<PositionIndex>(0), static_cast<PositionIndex>(2)] = 999.0f;
    scores[static_cast<PositionIndex>(0), static_cast<PositionIndex>(3)] = 999.0f;

    scores[static_cast<PositionIndex>(1), static_cast<PositionIndex>(0)] = 1.0f;
    scores[static_cast<PositionIndex>(1), static_cast<PositionIndex>(1)] = 2.0f;
    scores[static_cast<PositionIndex>(1), static_cast<PositionIndex>(2)] = 999.0f;
    scores[static_cast<PositionIndex>(1), static_cast<PositionIndex>(3)] = 999.0f;

    scores[static_cast<PositionIndex>(2), static_cast<PositionIndex>(0)] = 0.0f;
    scores[static_cast<PositionIndex>(2), static_cast<PositionIndex>(1)] = 1.0f;
    scores[static_cast<PositionIndex>(2), static_cast<PositionIndex>(2)] = 2.0f;
    scores[static_cast<PositionIndex>(2), static_cast<PositionIndex>(3)] = 999.0f;

    scores[static_cast<PositionIndex>(3), static_cast<PositionIndex>(0)] = -1.0f;
    scores[static_cast<PositionIndex>(3), static_cast<PositionIndex>(1)] = 0.0f;
    scores[static_cast<PositionIndex>(3), static_cast<PositionIndex>(2)] = 1.0f;
    scores[static_cast<PositionIndex>(3), static_cast<PositionIndex>(3)] = 2.0f;

    TransformerBlock::causal_softmax_for_test(scores, T);

    // Row 0: only self is allowed.
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(0), static_cast<PositionIndex>(0)), 1.0f, 1e-6f);

    // Row 1 expected softmax([1,2]).
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(1), static_cast<PositionIndex>(0)), 0.26894143f, 1e-6f);
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(1), static_cast<PositionIndex>(1)), 0.73105860f, 1e-6f);

    // Row 2 expected softmax([0,1,2]).
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(0)), 0.09003057f, 1e-6f);
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(1)), 0.24472848f, 1e-6f);
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(2)), 0.66524094f, 1e-6f);

    // Row 3 expected softmax([-1,0,1,2]).
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(0)), 0.03205860f, 1e-6f);
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(1)), 0.08714432f, 1e-6f);
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(2)), 0.23688284f, 1e-6f);
    EXPECT_NEAR(scores.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(3)), 0.64391428f, 1e-6f);

    // Row-wise normalization over active (causal) region and strict masking for j > i.
    for (int i = 0; i < 4; ++i)
    {
        float row_sum = 0.0f;
        for (int j = 0; j <= i; ++j)
            row_sum += scores.get(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j));
        EXPECT_NEAR(row_sum, 1.0f, 1e-6f);

        for (int j = i + 1; j < 4; ++j)
            EXPECT_FLOAT_EQ(scores.get(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j)), 0.0f);
    }
}

TEST(TransformerBlockTest, SoftmaxBackwardMatchesJacobianAndAccumulates)
{
    using namespace rllm;

    const auto T = static_cast<PositionIndex>(4);
    flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex> dp(T, T);
    flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex> p(T, T);
    flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex> dscores(T, T);

    dp.fill(0.0f);
    p.fill(0.0f);
    dscores.fill(0.1f); // Verify accumulation semantics: dscores += ...

    // Row 0 (single active element)
    p.set(static_cast<PositionIndex>(0), static_cast<PositionIndex>(0), 1.0f);
    dp.set(static_cast<PositionIndex>(0), static_cast<PositionIndex>(0), 3.0f);

    // Row 1: active j in [0,1], p sums to 1
    p.set(static_cast<PositionIndex>(1), static_cast<PositionIndex>(0), 0.2f);
    p.set(static_cast<PositionIndex>(1), static_cast<PositionIndex>(1), 0.8f);
    dp.set(static_cast<PositionIndex>(1), static_cast<PositionIndex>(0), 1.0f);
    dp.set(static_cast<PositionIndex>(1), static_cast<PositionIndex>(1), -2.0f);

    // Row 2: active j in [0,2], p sums to 1
    p.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(0), 0.1f);
    p.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(1), 0.3f);
    p.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(2), 0.6f);
    dp.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(0), 2.0f);
    dp.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(1), -1.0f);
    dp.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(2), 0.5f);

    // Row 3: active j in [0,3], p sums to 1
    p.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(0), 0.25f);
    p.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(1), 0.25f);
    p.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(2), 0.25f);
    p.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(3), 0.25f);
    dp.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(0), -1.0f);
    dp.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(1), 0.0f);
    dp.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(2), 1.0f);
    dp.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(3), 2.0f);

    TransformerBlock::softmax_backward_for_test(dp, p, dscores, T);

    // Expected updates per row: p_j * (dp_j - dot), where dot = sum_k dp_k * p_k.
    // Row 0: dot=3, update=[0]
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(0), static_cast<PositionIndex>(0)), 0.1f, 1e-6f);

    // Row 1: dot = 1*0.2 + (-2)*0.8 = -1.4
    // updates: j0=0.2*(1+1.4)=0.48, j1=0.8*(-2+1.4)=-0.48
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(1), static_cast<PositionIndex>(0)), 0.58f, 1e-6f);
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(1), static_cast<PositionIndex>(1)), -0.38f, 1e-6f);

    // Row 2: dot = 2*0.1 + (-1)*0.3 + 0.5*0.6 = 0.2
    // updates: [0.18, -0.36, 0.18]
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(0)), 0.28f, 1e-6f);
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(1)), -0.26f, 1e-6f);
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(2)), 0.28f, 1e-6f);

    // Row 3: dot = (-1 + 0 + 1 + 2) * 0.25 = 0.5
    // updates: [-0.375, -0.125, 0.125, 0.375]
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(0)), -0.275f, 1e-6f);
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(1)), -0.025f, 1e-6f);
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(2)), 0.225f, 1e-6f);
    EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(3)), 0.475f, 1e-6f);

    // Entries outside the causal region are untouched (remain baseline 0.1f).
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            EXPECT_NEAR(dscores.get(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j)), 0.1f, 1e-6f);
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
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> h_template(
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
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> h_template(
        static_cast<rllm::PositionIndex>(T));
    h_template.fill(0.1f);
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> dout_template(
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
        rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
            static_cast<rllm::PositionIndex>(T));
        block->backward(dout_template, din, 0.01f);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel ---
    parallel::set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
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

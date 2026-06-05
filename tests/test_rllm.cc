#include <gtest/gtest.h>

#include <Corpus.hpp>
#include <NeuralNetwork.hpp>
#include <Statistics.hpp>
#include <TransformerBlock.hpp>
#include <parallel.hpp>
#include <enum_iterator.hpp>
#include <enum_iterator2D.hpp>

#include <vecmath.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <print>
#include <vector>

using rllm::rlmm_float;

namespace
{
    // Deletes all files in models/ before each individual test case.
    class CleanModelsListener : public ::testing::EmptyTestEventListener
    {
        void OnTestStart(const ::testing::TestInfo&) override
        {
            if (std::filesystem::exists("models"))
                for (const auto& entry : std::filesystem::directory_iterator("models"))
                    std::filesystem::remove(entry.path());
        }
    };
} // namespace

int main(int argc, char** argv)
{
#if defined(USE_VULKAN_OFFLOAD)
    setenv("RLLM_VULKAN_POOL_BYTES", "868435456", 0);
#endif
    parallel::init_parallel();
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::UnitTest::GetInstance()->listeners().Append(new CleanModelsListener());
    return RUN_ALL_TESTS();
}

TEST(PredictorTest, Placeholder)
{
    SUCCEED();
}

namespace
{
    constexpr int BENCH_ITERS = 20;
    constexpr int BENCH_FORWARD_ITERS = 20; // forward is fast; needs more iters for stable timing
    constexpr int BENCH_RMS_NORM_ITERS = 50;
    constexpr int TEST_SEQ_LEN = 8; // smoke tests
    constexpr int BENCH_SEQ_LEN = 64; // speedup benchmark needs more work

    std::unique_ptr<rllm::NeuralNetwork> train_guaranteed_model(rllm::Corpus& corpus, rllm::Statistics& stats)
    {
        std::srand(0);
        corpus.load_files_from_dir("training_data0");
        auto nn = std::make_unique<rllm::NeuralNetwork>(1, corpus, stats);
        nn->set_training_method(rllm::TrainingMethod::RANDOM_LINE_RANDOM_LEN);
        nn->train(false, 3, std::nullopt, std::nullopt);
        return nn;
    }

    std::vector<rllm::OutputToken>
    top5_for_prompt(rllm::NeuralNetwork& nn, rllm::Corpus& corpus, const std::string& prompt)
    {
        const auto token_ids = corpus.get_token_ids(prompt);

        nn.get_last_input() = token_ids; // set the input to the probe token(s) for tracing

        nn.propagate_forward();
        const auto top5 = nn.get_best_output_token_ids(5, rllm::MultiTokenPredictionIndex::START);

        std::println("Prompt '{}', top-5:", prompt);
        for (size_t i = 0; i < top5.size(); ++i)
        {
            const auto tok = corpus.get_token_from_id(top5[i].token_id);
            std::println(
                "  [{}] token='{}' id={} p={:.6f}", i, tok, static_cast<int>(top5[i].token_id), top5[i].activation
            );
        }
        return top5;
    }
} // namespace

TEST(PredictorRegressionTest, GuaranteedModel_HashPredictsInclude)
{
    std::srand(0);
    // Dedicated corpus: 4 #include lines vs 1 #define line, all sharing the same
    // #include token sequence. Small enough (5 lines) that the validation split
    // stays below 2 lines, disabling early stopping and checkpoint restoration.
    std::vector<std::string> filters = {"include_sequence"};
    rllm::Corpus corpus(filters);
    corpus.load_files_from_dir("training_data0");
    rllm::Statistics stats;
    auto nn = std::make_unique<rllm::NeuralNetwork>(2, corpus, stats);
    nn->set_training_method(rllm::TrainingMethod::INCREASINGLY_LONGER_SEQUENCES);
    nn->train(false, 10, std::nullopt, std::nullopt);

    const auto top5 = top5_for_prompt(*nn, corpus, "#");
    ASSERT_FALSE(top5.empty());
    EXPECT_EQ(corpus.get_token_from_id(top5.front().token_id), "in");

    // MTP head 1: all output heads were already computed by the top5_for_prompt call above.
    // Head 1 should predict 'clu' — the 2nd token of '#include' — in parallel with head 0.
    {
        const auto& head1 = nn->get_output_layer(rllm::MultiTokenPredictionIndex::ONE);
        const auto head1_top = head1.get_top_k_by_logit(5);
        ASSERT_FALSE(head1_top.empty());
        std::println("Prompt '#', MTP head-1 top-5:");
        for (size_t i = 0; i < head1_top.size(); ++i)
        {
            const auto tok = corpus.get_token_from_id(head1_top[i].token_id);
            std::println(
                "  [{}] token='{}' id={} logit={:.6f}", i, tok, static_cast<int>(head1_top[i].token_id), head1_top[i].activation
            );
        }
        EXPECT_EQ(corpus.get_token_from_id(head1_top.front().token_id), "clu");
    }

    // MTP head 2: predict 'de' (3rd token of '#include') from context '#'.
    {
        const auto& head2 = nn->get_output_layer(rllm::MultiTokenPredictionIndex::TWO);
        const auto head2_top = head2.get_top_k_by_logit(5);
        ASSERT_FALSE(head2_top.empty());
        std::println("Prompt '#', MTP head-2 top-5:");
        for (size_t i = 0; i < head2_top.size(); ++i)
        {
            const auto tok = corpus.get_token_from_id(head2_top[i].token_id);
            std::println(
                "  [{}] token='{}' id={} logit={:.6f}", i, tok, static_cast<int>(head2_top[i].token_id), head2_top[i].activation
            );
        }
        EXPECT_EQ(corpus.get_token_from_id(head2_top.front().token_id), "de");
    }
}

// Focused MTP test: a single forward pass from '#' should predict 'in' on head 0
// and 'clu' on head 1 simultaneously — without any additional context tokens.
TEST(PredictorRegressionTest, MTP_HashPredictsInThenCluInParallel)
{
    std::srand(0);
    std::vector<std::string> filters = {"include_sequence"};
    rllm::Corpus corpus(filters);
    corpus.load_files_from_dir("training_data0");
    rllm::Statistics stats;
    auto nn = std::make_unique<rllm::NeuralNetwork>(2, corpus, stats);
    nn->set_training_method(rllm::TrainingMethod::INCREASINGLY_LONGER_SEQUENCES);
    nn->train(false, 10, std::nullopt, std::nullopt);

    // Single forward pass with '#' activates all MTP heads simultaneously.
    const auto hash_toks = corpus.get_token_ids("#");
    ASSERT_FALSE(hash_toks.empty());
    nn->get_last_input() = hash_toks;
    nn->propagate_forward();

    // Head 0 (primary): predicts the next token after '#'.
    const auto head0_top = nn->get_best_output_token_ids(1, rllm::MultiTokenPredictionIndex::START);
    ASSERT_FALSE(head0_top.empty());
    EXPECT_EQ(corpus.get_token_from_id(head0_top.front().token_id), "in")
        << "Head 0 should predict 'in' (next token after '#') from context '#'";

    // Head 1: predicts the 2nd-next token from '#' in parallel with head 0.
    const auto& head1 = nn->get_output_layer(rllm::MultiTokenPredictionIndex::ONE);
    const auto head1_top = head1.get_top_k_by_logit(1);
    ASSERT_FALSE(head1_top.empty());
    EXPECT_EQ(corpus.get_token_from_id(head1_top.front().token_id), "clu")
        << "Head 1 should predict 'clu' (2nd token of '#include') in parallel with head 0";
}


TEST(PredictorRegressionTest, GuaranteedModel_IncludePredictsA)
{
    std::srand(0);
    // Dedicated file: only "#include A" and "#define B".
    // With MTP (4 heads), "#include A" (5 tokens) trains from context "[#]":
    //   head 0→"in", head 1→"clu", head 2→"de", head 3→"A".
    // "#define B" (4 tokens) only activates 3 heads, so head 3 is exclusively
    // trained to predict "A" from "[#]" — no conflicting signal.
    std::vector<std::string> filters = {"include_a_training"};
    rllm::Corpus corpus(filters);
    corpus.load_files_from_dir("training_data0");
    rllm::Statistics stats;
    auto nn = std::make_unique<rllm::NeuralNetwork>(1, corpus, stats);
    nn->set_training_method(rllm::TrainingMethod::RANDOM_LINE_FULL);
    nn->train(false, 10, std::nullopt, std::nullopt);

    // Head 3 (THREE) should predict "A" — the 4th token after "#".
    const auto hash_toks = corpus.get_token_ids("#");
    ASSERT_FALSE(hash_toks.empty());

    nn->get_last_input() = hash_toks; // set the input to the probe token(s) for tracing
    nn->propagate_forward();
    const auto top1 = nn->get_best_output_token_ids(1, rllm::MultiTokenPredictionIndex::THREE);
    ASSERT_FALSE(top1.empty());
    EXPECT_EQ(corpus.get_token_from_id(top1.front().token_id), "A")
        << "Expected MTP head 3 to predict 'A' (4th token of '#include A') from context '#'";
}

TEST(PredictorRegressionTest, SimplestGuaranteedTraining_HashKeepsDefineAboveFloor)
{
    // Keep this training deterministic inside the test process.
    std::srand(0);

    std::vector<std::string> filters = {"guaranteed_to_learn"};
    rllm::Corpus corpus(filters);
    corpus.load_files_from_dir("training_data0");
    rllm::Statistics stats;
    auto nn = std::make_unique<rllm::NeuralNetwork>(1, corpus, stats);
    nn->set_training_method(rllm::TrainingMethod::RANDOM_LINE_RANDOM_LEN);

    // 3 epochs is the smallest fast setting that consistently pushes
    // '# -> defin' above the old ~0.6% floor on this tiny corpus.
    nn->train(false, 3, std::nullopt, std::nullopt);

    const auto prompt = corpus.get_token_ids("#");
    nn->get_last_input() = prompt;
    nn->propagate_forward();
    const auto top5 = nn->get_best_output_token_ids(5, rllm::MultiTokenPredictionIndex::START);
    ASSERT_EQ(top5.size(), 5u);

    bool include_seen = false;
    bool defin_seen = false;
    float defin_probability = 0.0f;

    int i = 0;
    for (const auto& out : top5)
    {
        const auto token = corpus.get_token_from_id(out.token_id);

        std::println("Top-5 token[{}]: '{}' with probability {:.6f}", i, token, out.activation);
        ++i;

        if (token == "in")
            include_seen = true;
        if (token == "defin")
        {
            defin_seen = true;
            defin_probability = out.activation;
        }
    }

    EXPECT_TRUE(include_seen) << "Expected 'inclu' in top-5 for prompt '#'";
    EXPECT_TRUE(defin_seen) << "Expected 'defin' in top-5 for prompt '#'";
    EXPECT_GT(defin_probability, 0.006f) << "Expected 'defin' probability > 0.6%, got " << (defin_probability * 100.0f)
                                         << "%";
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
        static_cast<rllm::PositionIndex>(T)
    );
    fill(h, 0.1f);

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
        static_cast<rllm::PositionIndex>(T)
    );
    fill(h, 0.05f);
    block->forward(h, static_cast<rllm::PositionIndex>(T));

    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> dout(
        static_cast<rllm::PositionIndex>(T)
    );
    fill(dout, 0.01f);
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
        static_cast<rllm::PositionIndex>(T)
    );
    rllm::BackwardWorkspace backward_workspace(static_cast<rllm::PositionIndex>(T));
    block->backward(dout, din, backward_workspace, 0.01f);

    ASSERT_EQ(static_cast<int>(din.num_rows()) * static_cast<int>(din.num_cols()), T * D)
        << "backward() must return a gradient of the same size as the input";
}

TEST(TransformerBlockTest, RmsNormBenchmark)
{
    const int T = BENCH_SEQ_LEN;
    const int D = static_cast<int>(rllm::EmbeddingDimension::MAX);
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> x(
        static_cast<rllm::PositionIndex>(T)
    );
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> y(
        static_cast<rllm::PositionIndex>(T)
    );
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> y_vulkan_optimized(
        static_cast<rllm::PositionIndex>(T)
    );

    for (int t = 0; t < T; ++t)
    {
        for (int i = 0; i < D; ++i)
            x[t, i] = 0.01f * static_cast<float>((t % 7) + 1) + 0.001f * static_cast<float>((i % 31) + 1);
    }

    rllm::TransformerBlock::apply_rms_norm(x, y);
    rllm::TransformerBlock::apply_rms_norm_vulkan_optimized(x, y_vulkan_optimized);

    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_RMS_NORM_ITERS; ++iter)
        rllm::TransformerBlock::apply_rms_norm(x, y);
    const auto t1 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_RMS_NORM_ITERS; ++iter)
        rllm::TransformerBlock::apply_rms_norm_vulkan_optimized(x, y_vulkan_optimized);
    const auto t2 = std::chrono::steady_clock::now();

    const auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto vulkan_optimized_total_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const double per_iter_us = static_cast<double>(total_us) / static_cast<double>(BENCH_RMS_NORM_ITERS);
    const double vulkan_optimized_per_iter_us =
        static_cast<double>(vulkan_optimized_total_us) / static_cast<double>(BENCH_RMS_NORM_ITERS);

    RecordProperty("rms_norm_total_us", total_us);
    RecordProperty("rms_norm_per_iter_us", per_iter_us);
    RecordProperty("rms_norm_vulkan_optimized_total_us", vulkan_optimized_total_us);
    RecordProperty("rms_norm_vulkan_optimized_per_iter_us", vulkan_optimized_per_iter_us);
    fprintf(
        stdout,
        "TransformerBlock RMSNorm - %d iters: baseline=%lldus total %.2fus/iter, vulkan_optimized=%lldus total %.2fus/iter\n",
        BENCH_RMS_NORM_ITERS,
        static_cast<long long>(total_us),
        per_iter_us,
        static_cast<long long>(vulkan_optimized_total_us),
        vulkan_optimized_per_iter_us
    );

    const rlmm_float* y_data = y.data();
    const rlmm_float* y_vulkan_optimized_data = y_vulkan_optimized.data();
    for (int t = 0; t < T; ++t)
    {
        double sq = 0.0;
        for (int i = 0; i < D; ++i)
        {
            const float value = y_data[t * D + i];
            ASSERT_TRUE(std::isfinite(value));
            sq += static_cast<double>(value) * static_cast<double>(value);
            EXPECT_NEAR(y_vulkan_optimized_data[t * D + i], value, 2e-4f);
        }
        EXPECT_NEAR(sq / static_cast<double>(D), 1.0, 2e-3);
    }
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

    dp.zero();
    p.zero();
    fill(dscores, 0.1f); // Verify accumulation semantics: dscores += ...

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
        static_cast<rllm::PositionIndex>(T)
    );
    fill(h_template, 0.1f);

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_FORWARD_ITERS; ++iter)
    {
        std::println("Forward seq iter {}/{}", iter + 1, BENCH_FORWARD_ITERS);
        auto h = h_template;
        block->forward(h, static_cast<rllm::PositionIndex>(T));
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel ---
    parallel::set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_FORWARD_ITERS; ++iter)
    {
        std::println("Forward par iter {}/{}", iter + 1, BENCH_FORWARD_ITERS);
        auto h = h_template;
        block->forward(h, static_cast<rllm::PositionIndex>(T));
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us", serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup", speedup);

    fprintf(
        stdout,
        "TransformerBlock Forward - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
        static_cast<long long>(serial_us),
        static_cast<long long>(parallel_us),
        speedup
    );

    EXPECT_GT(speedup, 1.0) << "parallelisation of TransformerBlock::forward is slower than serial "
                            << "(serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup
                            << ").";
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
        static_cast<rllm::PositionIndex>(T)
    );
    fill(h_template, 0.1f);
    rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> dout_template(
        static_cast<rllm::PositionIndex>(T)
    );
    fill(dout_template, 0.01f);

    // Prime the block with a forward pass so backward has valid cached state.
    {
        auto h = h_template;
        block->forward(h, static_cast<rllm::PositionIndex>(T));
    }

    // --- serial baseline (1 thread) ---
    rllm::BackwardWorkspace backward_workspace(static_cast<rllm::PositionIndex>(T));
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        std::print("Backward iter {}/{}\n", iter + 1, BENCH_ITERS);
        rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
            static_cast<rllm::PositionIndex>(T)
        );
        block->backward(dout_template, din, backward_workspace, 0.01f);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel ---
    parallel::set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        std::println("Backward par iter {}/{}", iter + 1, BENCH_ITERS);
        rllm::flexible_rows_matrix<rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> din(
            static_cast<rllm::PositionIndex>(T)
        );
        block->backward(dout_template, din, backward_workspace, 0.01f);
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us", serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup", speedup);

    fprintf(
        stdout,
        "TransformerBlock Backward - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
        static_cast<long long>(serial_us),
        static_cast<long long>(parallel_us),
        speedup
    );
    PARALLEL_DUMP_STATS();

    EXPECT_GT(speedup, 1.0) << "parallelisation of TransformerBlock::backward is slower than serial "
                            << "(serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup
                            << ").";
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

    constexpr int ROWS = BENCH_SEQ_LEN; // 64
    constexpr int COLS = static_cast<int>(rllm::EmbeddingDimension::MAX); // 512
    constexpr int WORK_PER_CELL = 128; // float multiply-adds per cell
    constexpr int ITERS = 100;

    const auto seq = static_cast<rllm::PositionIndex>(ROWS);
    std::vector<float> buf(ROWS * COLS, 0.0f);

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it)
    {
        for (const auto [t, d] : rllm::enum_iterator2D<rllm::PositionIndex, rllm::EmbeddingDimension>(seq))
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

    const auto serial_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us", serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup", speedup);

    fprintf(
        stderr,
        "PARFOR_2D (%d outer x %d inner, work=%d, iters=%d, threads=%d)"
        " - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
        ROWS,
        COLS,
        WORK_PER_CELL,
        ITERS,
        max_threads,
        static_cast<long long>(serial_us),
        static_cast<long long>(parallel_us),
        speedup
    );

    EXPECT_GT(speedup, 1.0) << "PARFOR_2D was not faster than serial"
                            << " (serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup
                            << ").";
}

// ── PARFOR_2D_TRIANGULAR tests ────────────────────────────────────────────────

// Verifies that PARFOR_2D_TRIANGULAR visits every (i,j) pair with j <= i
// exactly once and never touches pairs above the diagonal.
// Parallelism is over the outer i; j iterates sequentially within each task,
// so writing to count[i][...] from different tasks (different i) is race-free.
TEST(Parfor2DTriangularTest, VisitsAllLowerTriangularPairsExactlyOnce)
{
    using namespace rllm;
    constexpr int N = 8;
    const auto N_pos = static_cast<PositionIndex>(N);

    int count[N][N] = {};

    PARFOR_2D_TRIANGULAR(i, j, N_pos)
    ++count[static_cast<int>(i)][static_cast<int>(j)];
    ENDFOR

    int total = 0;
    for (int ii = 0; ii < N; ++ii)
    {
        for (int jj = 0; jj < N; ++jj)
        {
            const int expected = (jj <= ii) ? 1 : 0;
            EXPECT_EQ(count[ii][jj], expected) << "Wrong visit count at (" << ii << "," << jj << ")";
            total += count[ii][jj];
        }
    }
    EXPECT_EQ(total, N * (N + 1) / 2);
}

// Verifies that PARFOR_2D_TRIANGULAR produces a parallel speedup over serial
// execution on a compute-bound workload.
TEST(Parfor2DTriangularTest, SpeedupFasterThanSerial)
{
    if (parallel::get_max_threads() < 2)
        GTEST_SKIP() << "thread count < 2 - no parallelism available";

    const int max_threads = parallel::get_max_threads();

    constexpr int N = BENCH_SEQ_LEN; // 64 rows → 2080 triangular cells
    constexpr int WORK_PER_CELL = 256; // float multiply-adds per cell
    constexpr int ITERS = 100;

    const auto N_pos = static_cast<rllm::PositionIndex>(N);
    std::vector<float> buf(static_cast<size_t>(N) * static_cast<size_t>(N), 0.0f);

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it)
    {
        for (int ii = 0; ii < N; ++ii)
            for (int jj = 0; jj <= ii; ++jj)
            {
                float v = static_cast<float>(ii * N + jj + it);
                for (int k = 0; k < WORK_PER_CELL; ++k)
                    v = v * 1.00001f + 0.00001f;
                buf[static_cast<size_t>(ii) * N + static_cast<size_t>(jj)] = v;
            }
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel (PARFOR_2D_TRIANGULAR) ---
    parallel::set_num_threads(max_threads);
    std::fill(buf.begin(), buf.end(), 0.0f);
    const auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it)
    {
        PARFOR_2D_TRIANGULAR(i, j, N_pos)
        const int ii = static_cast<int>(i);
        const int jj = static_cast<int>(j);
        float v = static_cast<float>(ii * N + jj + it);
        for (int k = 0; k < WORK_PER_CELL; ++k)
            v = v * 1.00001f + 0.00001f;
        buf[static_cast<size_t>(ii) * N + static_cast<size_t>(jj)] = v;
        ENDFOR
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us", serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup", speedup);

    fprintf(
        stderr,
        "PARFOR_2D_TRIANGULAR (%d rows, %d triangular cells, work=%d, iters=%d, threads=%d)"
        " - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
        N,
        N * (N + 1) / 2,
        WORK_PER_CELL,
        ITERS,
        max_threads,
        static_cast<long long>(serial_us),
        static_cast<long long>(parallel_us),
        speedup
    );

    EXPECT_GT(speedup, 1.0) << "PARFOR_2D_TRIANGULAR was not faster than serial"
                            << " (serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup
                            << ").";
}

// ── PARFOR_2D_UPPER_TRIANGULAR tests ─────────────────────────────────────────

// Verifies that PARFOR_2D_UPPER_TRIANGULAR visits every (i,j) pair with j >= i
// exactly once and never touches pairs below the diagonal.
// Parallelism is over the outer i; j iterates sequentially within each task,
// so writing to count[i][...] from different tasks (different i) is race-free.
TEST(Parfor2DUpperTriangularTest, VisitsAllUpperTriangularPairsExactlyOnce)
{
    using namespace rllm;
    constexpr int N = 8;
    const auto N_pos = static_cast<PositionIndex>(N);

    int count[N][N] = {};

    PARFOR_2D_UPPER_TRIANGULAR(i, j, N_pos)
    ++count[static_cast<int>(i)][static_cast<int>(j)];
    ENDFOR

    int total = 0;
    for (int ii = 0; ii < N; ++ii)
    {
        for (int jj = 0; jj < N; ++jj)
        {
            const int expected = (jj >= ii) ? 1 : 0;
            EXPECT_EQ(count[ii][jj], expected) << "Wrong visit count at (" << ii << "," << jj << ")";
            total += count[ii][jj];
        }
    }
    EXPECT_EQ(total, N * (N + 1) / 2);
}

// Verifies that PARFOR_2D_UPPER_TRIANGULAR produces a parallel speedup over
// serial execution on a compute-bound workload.
TEST(Parfor2DUpperTriangularTest, SpeedupFasterThanSerial)
{
    if (parallel::get_max_threads() < 2)
        GTEST_SKIP() << "thread count < 2 - no parallelism available";

    const int max_threads = parallel::get_max_threads();

    constexpr int N = BENCH_SEQ_LEN; // 64 rows → 2080 upper-triangular cells
    constexpr int WORK_PER_CELL = 256;
    constexpr int ITERS = 100;

    const auto N_pos = static_cast<rllm::PositionIndex>(N);
    std::vector<float> buf(static_cast<size_t>(N) * static_cast<size_t>(N), 0.0f);

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it)
    {
        for (int ii = 0; ii < N; ++ii)
            for (int jj = ii; jj < N; ++jj)
            {
                float v = static_cast<float>(ii * N + jj + it);
                for (int k = 0; k < WORK_PER_CELL; ++k)
                    v = v * 1.00001f + 0.00001f;
                buf[static_cast<size_t>(ii) * N + static_cast<size_t>(jj)] = v;
            }
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel (PARFOR_2D_UPPER_TRIANGULAR) ---
    parallel::set_num_threads(max_threads);
    std::fill(buf.begin(), buf.end(), 0.0f);
    const auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it)
    {
        PARFOR_2D_UPPER_TRIANGULAR(i, j, N_pos)
        const int ii = static_cast<int>(i);
        const int jj = static_cast<int>(j);
        float v = static_cast<float>(ii * N + jj + it);
        for (int k = 0; k < WORK_PER_CELL; ++k)
            v = v * 1.00001f + 0.00001f;
        buf[static_cast<size_t>(ii) * N + static_cast<size_t>(jj)] = v;
        ENDFOR
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us", serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup", speedup);

    fprintf(
        stderr,
        "PARFOR_2D_UPPER_TRIANGULAR (%d rows, %d upper-triangular cells, work=%d, iters=%d, threads=%d)"
        " - Serial: %lldus, Parallel: %lldus, Speedup: %.2fx\n",
        N,
        N * (N + 1) / 2,
        WORK_PER_CELL,
        ITERS,
        max_threads,
        static_cast<long long>(serial_us),
        static_cast<long long>(parallel_us),
        speedup
    );

    EXPECT_GT(speedup, 1.0) << "PARFOR_2D_UPPER_TRIANGULAR was not faster than serial"
                            << " (serial=" << serial_us << "us, parallel=" << parallel_us << "us, speedup=" << speedup
                            << ").";
}

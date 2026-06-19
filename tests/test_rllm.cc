#include <gtest/gtest.h>

#include <Corpus.hpp>
#include <NeuralNetwork.hpp>
#include <Statistics.hpp>
#include <TransformerBlock.hpp>
#include <parallel.hpp>
#include <enum_iterator1D.hpp>
#include <enum_iterator2D.hpp>

#include <vecmath.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <print>
#include <vector>

using namespace rlmm;

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

    std::unique_ptr<NeuralNetwork> train_guaranteed_model(Corpus& corpus, Statistics& stats)
    {
        std::srand(0);
        corpus.load_files_from_dir("training_data0");
        auto nn = std::make_unique<NeuralNetwork>(1, corpus, stats);
        nn->set_training_method(TrainingMethod::RANDOM_LINE_RANDOM_LEN);
        nn->train(false, 3, std::nullopt, std::nullopt);
        return nn;
    }

    std::vector<OutputToken>
    top5_for_prompt(NeuralNetwork& nn, Corpus& corpus, const std::string& prompt)
    {
        const auto token_ids = corpus.get_token_ids(prompt);

        nn.get_last_input() = token_ids; // set the input to the probe token(s) for tracing

        nn.propagate_forward();
        const auto top5 = nn.get_best_output_token_ids(5, MultiTokenPredictionIndex::START);

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
    Corpus corpus(filters);
    corpus.load_files_from_dir("training_data0");
    Statistics stats;
    auto nn = std::make_unique<NeuralNetwork>(2, corpus, stats);
    nn->set_training_method(TrainingMethod::INCREASINGLY_LONGER_SEQUENCES);
    nn->train(false, 10, std::nullopt, std::nullopt);

    const auto top5 = top5_for_prompt(*nn, corpus, "#");
    ASSERT_FALSE(top5.empty());
    EXPECT_EQ(corpus.get_token_from_id(top5.front().token_id), "in");

    // MTP head 1: all output heads were already computed by the top5_for_prompt call above.
    // Head 1 should predict 'clu' — the 2nd token of '#include' — in parallel with head 0.
    {
        const auto& head1 = nn->get_output_layer(MultiTokenPredictionIndex::ONE);
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
        const auto& head2 = nn->get_output_layer(MultiTokenPredictionIndex::TWO);
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
    Corpus corpus(filters);
    corpus.load_files_from_dir("training_data0");
    Statistics stats;
    auto nn = std::make_unique<NeuralNetwork>(2, corpus, stats);
    nn->set_training_method(TrainingMethod::INCREASINGLY_LONGER_SEQUENCES);
    nn->train(false, 10, std::nullopt, std::nullopt);

    // Single forward pass with '#' activates all MTP heads simultaneously.
    const auto hash_toks = corpus.get_token_ids("#");
    ASSERT_FALSE(hash_toks.empty());
    nn->get_last_input() = hash_toks;
    nn->propagate_forward();

    // Head 0 (primary): predicts the next token after '#'.
    const auto head0_top = nn->get_best_output_token_ids(1, MultiTokenPredictionIndex::START);
    ASSERT_FALSE(head0_top.empty());
    EXPECT_EQ(corpus.get_token_from_id(head0_top.front().token_id), "in")
        << "Head 0 should predict 'in' (next token after '#') from context '#'";

    // Head 1: predicts the 2nd-next token from '#' in parallel with head 0.
    const auto& head1 = nn->get_output_layer(MultiTokenPredictionIndex::ONE);
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
    Corpus corpus(filters);
    corpus.load_files_from_dir("training_data0");
    Statistics stats;
    auto nn = std::make_unique<NeuralNetwork>(1, corpus, stats);
    nn->set_training_method(TrainingMethod::RANDOM_LINE_FULL);
    nn->train(false, 10, std::nullopt, std::nullopt);

    // Head 3 (THREE) should predict "A" — the 4th token after "#".
    const auto hash_toks = corpus.get_token_ids("#");
    ASSERT_FALSE(hash_toks.empty());

    nn->get_last_input() = hash_toks; // set the input to the probe token(s) for tracing
    nn->propagate_forward();
    const auto top1 = nn->get_best_output_token_ids(1, MultiTokenPredictionIndex::THREE);
    ASSERT_FALSE(top1.empty());
    EXPECT_EQ(corpus.get_token_from_id(top1.front().token_id), "A")
        << "Expected MTP head 3 to predict 'A' (4th token of '#include A') from context '#'";
}

TEST(PredictorRegressionTest, SimplestGuaranteedTraining_HashKeepsDefineAboveFloor)
{
    // Keep this training deterministic inside the test process.
    std::srand(0);

    std::vector<std::string> filters = {"guaranteed_to_learn"};
    Corpus corpus(filters);
    corpus.load_files_from_dir("training_data0");
    Statistics stats;
    auto nn = std::make_unique<NeuralNetwork>(1, corpus, stats);
    nn->set_training_method(TrainingMethod::RANDOM_LINE_RANDOM_LEN);

    // 3 epochs is the smallest fast setting that consistently keeps both
    // preprocessor branches visible on this tiny corpus.
    nn->train(false, 3, std::nullopt, std::nullopt);

    const auto prompt = corpus.get_token_ids("#");
    const auto defin_tokens = corpus.get_token_ids("defin");
    ASSERT_EQ(defin_tokens.size(), static_cast<PositionIndex>(1));
    const auto defin_token_id = defin_tokens[PositionIndex::START];
    nn->get_last_input() = prompt;
    nn->propagate_forward();
    const auto top5 = nn->get_best_output_token_ids(5, MultiTokenPredictionIndex::START);
    const auto all_outputs = nn->get_best_output_token_ids(static_cast<size_t>(TokenID::MAX), MultiTokenPredictionIndex::START);
    ASSERT_EQ(top5.size(), 5u);

    bool include_seen = false;
    float defin_probability = 0.0f;

    int i = 0;
    for (const auto& out : top5)
    {
        const auto token = corpus.get_token_from_id(out.token_id);

        std::println("Top-5 token[{}]: '{}' with probability {:.6f}", i, token, out.activation);
        ++i;

        if (token == "in")
            include_seen = true;
    }

    for (const auto& out : all_outputs)
    {
        if (out.token_id == defin_token_id)
        {
            defin_probability = out.activation;
            break;
        }
    }

    EXPECT_TRUE(include_seen) << "Expected 'inclu' in top-5 for prompt '#'";
    EXPECT_GT(defin_probability, 0.001f) << "Expected 'defin' probability > 0.1%, got " << (defin_probability * 100.0f)
                                         << "%";
}

// ---------------------------------------------------------------------------
// TransformerBlock smoke test: forward produces output of correct shape
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, ForwardOutputShape)
{
    auto block = std::make_unique<TransformerBlock>();
    block->randomize();

    const int T = TEST_SEQ_LEN;
    const int D = static_cast<int>(EmbeddingDimension::MAX);
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h(
        static_cast<PositionIndex>(T)
    );
    fill(h, 0.1f);

    auto forward_workspace = std::make_unique<ForwardWorkspace>(static_cast<PositionIndex>(T));
    block->forward(h, static_cast<PositionIndex>(T), *forward_workspace);

    ASSERT_EQ(static_cast<size_t>(h.num_rows()), static_cast<size_t>(T));
}

// ---------------------------------------------------------------------------
// TransformerBlock smoke test: backward runs and returns correct shape
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, BackwardOutputShape)
{
    auto block = std::make_unique<TransformerBlock>();
    block->randomize();

    const int T = TEST_SEQ_LEN;
    const int D = static_cast<int>(EmbeddingDimension::MAX);
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h(
        static_cast<PositionIndex>(T)
    );
    fill(h, 0.05f);
    auto forward_workspace = std::make_unique<ForwardWorkspace>(static_cast<PositionIndex>(T));
    block->forward(h, static_cast<PositionIndex>(T), *forward_workspace);

    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> dout(
        static_cast<PositionIndex>(T)
    );
    fill(dout, 0.01f);
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> din(
        static_cast<PositionIndex>(T)
    );
    auto backward_workspace = std::make_unique<BackwardWorkspace>(static_cast<PositionIndex>(T));
    block->backward(dout, din, *backward_workspace, 0.01f, *forward_workspace);

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

TEST(TransformerBlockTest, SoftmaxAttentionForHeadMatchesJacobian)
{
    using namespace rllm;

    const auto T = static_cast<PositionIndex>(4);
    fixed_size_triangular_matrix<rlmm_float, PositionIndex, PositionIndex> d_scores;
    fixed_size_triangular_matrix<rlmm_float, PositionIndex, PositionIndex> attn_w;
    fixed_size_triangular_matrix<rlmm_float, PositionIndex, PositionIndex> d_raw;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j <= i; ++j)
        {
            d_scores.set(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j), 0.0f);
            attn_w.set(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j), 0.0f);
            d_raw.set(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j), -7.0f);
        }

    // Row 0 (single active element)
    attn_w.set(static_cast<PositionIndex>(0), static_cast<PositionIndex>(0), 1.0f);
    d_scores.set(static_cast<PositionIndex>(0), static_cast<PositionIndex>(0), 3.0f);

    // Row 1: active j in [0,1], p sums to 1
    attn_w.set(static_cast<PositionIndex>(1), static_cast<PositionIndex>(0), 0.2f);
    attn_w.set(static_cast<PositionIndex>(1), static_cast<PositionIndex>(1), 0.8f);
    d_scores.set(static_cast<PositionIndex>(1), static_cast<PositionIndex>(0), 1.0f);
    d_scores.set(static_cast<PositionIndex>(1), static_cast<PositionIndex>(1), -2.0f);

    // Row 2: active j in [0,2], p sums to 1
    attn_w.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(0), 0.1f);
    attn_w.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(1), 0.3f);
    attn_w.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(2), 0.6f);
    d_scores.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(0), 2.0f);
    d_scores.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(1), -1.0f);
    d_scores.set(static_cast<PositionIndex>(2), static_cast<PositionIndex>(2), 0.5f);

    // Row 3: active j in [0,3], p sums to 1
    attn_w.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(0), 0.25f);
    attn_w.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(1), 0.25f);
    attn_w.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(2), 0.25f);
    attn_w.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(3), 0.25f);
    d_scores.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(0), -1.0f);
    d_scores.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(1), 0.0f);
    d_scores.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(2), 1.0f);
    d_scores.set(static_cast<PositionIndex>(3), static_cast<PositionIndex>(3), 2.0f);

    TransformerBlock::softmax_attention_for_head_for_test(d_scores, d_raw, attn_w, T);

    // Expected updates per row: p_j * (dp_j - dot), where dot = sum_k dp_k * p_k.
    // Row 0: dot=3, update=[0]
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(0), static_cast<PositionIndex>(0)), 0.0f, 1e-6f);

    // Row 1: dot = 1*0.2 + (-2)*0.8 = -1.4
    // updates: j0=0.2*(1+1.4)=0.48, j1=0.8*(-2+1.4)=-0.48
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(1), static_cast<PositionIndex>(0)), 0.48f, 1e-6f);
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(1), static_cast<PositionIndex>(1)), -0.48f, 1e-6f);

    // Row 2: dot = 2*0.1 + (-1)*0.3 + 0.5*0.6 = 0.2
    // updates: [0.18, -0.36, 0.18]
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(0)), 0.18f, 1e-6f);
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(1)), -0.36f, 1e-6f);
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(2), static_cast<PositionIndex>(2)), 0.18f, 1e-6f);

    // Row 3: dot = (-1 + 0 + 1 + 2) * 0.25 = 0.5
    // updates: [-0.375, -0.125, 0.125, 0.375]
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(0)), -0.375f, 1e-6f);
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(1)), -0.125f, 1e-6f);
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(2)), 0.125f, 1e-6f);
    EXPECT_NEAR(d_raw.get(static_cast<PositionIndex>(3), static_cast<PositionIndex>(3)), 0.375f, 1e-6f);
}

// ---------------------------------------------------------------------------
// TransformerBlock test: parallel forward faster than serial
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, ForwardParallelFasterThanSerial)
{
    if (parallel::get_max_threads() < 2)
        GTEST_SKIP() << "thread count < 2 - no parallelism available";

    const int max_threads = parallel::get_max_threads();

    auto block = std::make_unique<TransformerBlock>();
    block->randomize();

    const int T = BENCH_SEQ_LEN;
    const int D = static_cast<int>(EmbeddingDimension::MAX);
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_template(
        static_cast<PositionIndex>(T)
    );
    fill(h_template, 0.1f);

    ForwardWorkspace forward_workspace(static_cast<PositionIndex>(T));

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_FORWARD_ITERS; ++iter)
    {
        std::println("Forward seq iter {}/{}", iter + 1, BENCH_FORWARD_ITERS);
        auto h = h_template;
        block->forward(h, static_cast<PositionIndex>(T), forward_workspace);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel ---
    parallel::set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_FORWARD_ITERS; ++iter)
    {
        std::println("Forward par iter {}/{}", iter + 1, BENCH_FORWARD_ITERS);
        auto h = h_template;
        block->forward(h, static_cast<PositionIndex>(T), forward_workspace);
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

    auto block = std::make_unique<TransformerBlock>();
    block->randomize();
    
    const int T = BENCH_SEQ_LEN;


    auto forward_workspace = std::make_unique<ForwardWorkspace>(static_cast<PositionIndex>(BENCH_SEQ_LEN));
    auto backward_workspace = std::make_unique<BackwardWorkspace>(static_cast<PositionIndex>(T));

    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_template(
        static_cast<PositionIndex>(T)
    );
    fill(h_template, 0.1f);
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> dout_template(
        static_cast<PositionIndex>(T)
    );
    fill(dout_template, 0.01f);

    // Prime the block with a forward pass so backward has valid cached state.
    {
        auto h = h_template;
        block->forward(h, static_cast<PositionIndex>(T), *forward_workspace);
    }

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        std::print("Backward iter {}/{}\n", iter + 1, BENCH_ITERS);
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> din(
            static_cast<PositionIndex>(T)
        );
        block->backward(dout_template, din, *backward_workspace, 0.01f, *forward_workspace);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- parallel ---
    parallel::set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        std::println("Backward par iter {}/{}", iter + 1, BENCH_ITERS);
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> din(
            static_cast<PositionIndex>(T)
        );
        block->backward(dout_template, din, *backward_workspace, 0.01f, *forward_workspace);
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
    constexpr int COLS = static_cast<int>(EmbeddingDimension::MAX); // 512
    constexpr int WORK_PER_CELL = 128; // float multiply-adds per cell
    constexpr int ITERS = 100;

    const auto seq = static_cast<PositionIndex>(ROWS);
    std::vector<float> buf(ROWS * COLS, 0.0f);

    // --- serial baseline (1 thread) ---
    parallel::set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it)
    {
        for (const auto [t, d] : enum_iterator2D<PositionIndex, EmbeddingDimension>(seq))
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
        PARFOR_2D(t, d, enum_iterator2D<PositionIndex, EmbeddingDimension>(seq))
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

    const auto N_pos = static_cast<PositionIndex>(N);
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

    const auto N_pos = static_cast<PositionIndex>(N);
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


// ---------------------------------------------------------------------------
// TransformerBlock RMSNorm backward correctness vs. sequential CPU reference
// ---------------------------------------------------------------------------
TEST(TransformerBlockTest, RmsNormBackwardMatchesCpuReference)
{
    constexpr int T = 32;

    const auto seq   = static_cast<PositionIndex>(T);
    constexpr float eps = 1e-6f;
    constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);

    // Seed for reproducibility
    std::srand(42);

    // --- allocate dy, x, dx ---
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> dy(seq);
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> x(seq);
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> dx(seq);
    flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> dx_ref(seq);

    // Fill with random values in [-1, 1]
    for (const auto [t, d] : enum_iterator2D<PositionIndex, EmbeddingDimension>(seq)) {
        float v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
        dy[t, d] = v;
        x[t, d]  = v * 0.5f;           // smaller values for stability
        dx[t, d]  = 0.f;
        dx_ref[t, d] = 0.f;
    }

    // --- run the existing rms_norm_backward (parallel or not — we compare to ref) ---
    TransformerBlock::rms_norm_backward_for_test(dy, x, dx);

    // --- independent sequential CPU reference ---
    for (const auto t : enum_iterator1D<PositionIndex>(seq)) {
        float sq = 0.f;
        for (const auto d : enum_iterator1D<EmbeddingDimension>()) {
            sq += static_cast<float>(x[t, d]) * static_cast<float>(x[t, d]);
        }
        const float inv_rms = 1.0f / std::sqrt(sq / fd + eps);

        float dot = 0.f;
        for (const auto d : enum_iterator1D<EmbeddingDimension>()) {
            dot += static_cast<float>(dy[t, d]) * static_cast<float>(x[t, d]) * inv_rms;
        }
        dot /= fd;   // mean over embedding dimension

        for (const auto d : enum_iterator1D<EmbeddingDimension>()) {
            dx_ref[t, d] += inv_rms * (static_cast<float>(dy[t, d]) - static_cast<float>(x[t, d]) * inv_rms * dot);
        }
    }

    // --- compare ---
    float max_diff = 0.f;
    for (const auto [t, d] : enum_iterator2D<PositionIndex, EmbeddingDimension>(seq)) {
        const float diff = std::abs(static_cast<float>(dx[t, d]) - static_cast<float>(dx_ref[t, d]));
        if (diff > max_diff) max_diff = diff;
    }

    constexpr float tol = 1e-4f;
    EXPECT_LT(max_diff, tol)
        << "rms_norm_backward vs reference: max_diff=" << max_diff;
}

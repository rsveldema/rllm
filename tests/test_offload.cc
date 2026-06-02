#include <gtest/gtest.h>

#include <InputLayer.hpp>
#include <LayerPrimitives.hpp>
#include <device_pointer.hpp>
#include <enum_iterator.hpp>
#include <enum_iterator2D.hpp>
#include <fixed_size_matrix.hpp>
#include <fixed_size_vector.hpp>
#include <parallel.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#if defined(USE_VULKAN_OFFLOAD)
#define ATOMIC_INC(x) ((x)++)
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES = 1u;
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ = 0u;
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ = 1u;
#else
#define ATOMIC_INC(x) (++(x))
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES = 0u;
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ = 0u;
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ = 0u;
#endif

#if !defined(USE_OPENMP)

int main(int argc, char** argv)
{
    parallel::init_parallel();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class OffloadParForTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        parallel::statistics.reset_buffer_copy_counters();
    }
};

TEST_F(OffloadParForTest, OffloadParForVisitsEachIndexExactlyOnce)
{
    constexpr size_t N = 17;
    // OFFLOAD_PARAMETERS(visits)
    rllm::fixed_size_vector<int, rllm::PositionIndex> visits;
    visits.set_size(static_cast<rllm::PositionIndex>(N));
    // END_OFFLOAD_PARAMETERS

    OFFLOAD_PARFOR_1D_PARAM(i, rllm::enum_iterator<rllm::PositionIndex>(static_cast<rllm::PositionIndex>(N)), (visits))
    ATOMIC_INC(visits[static_cast<size_t>(i)]);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ);

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(visits[i], 1) << "Wrong visit count at i=" << i;

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ);
}

TEST_F(OffloadParForTest, OffloadParForParamVisitsEachIndexExactlyOnce)
{
    constexpr size_t N = 19;
    // PARFOR_PARAM requires the loop variable to be captured in a lambda, so we can't use a simple array of ints here.
    // OFFLOAD_PARAMETERS(visits)
    rllm::fixed_size_vector<int, rllm::PositionIndex> visits;
    visits.set_size(static_cast<rllm::PositionIndex>(N));
    // END_OFFLOAD_PARAMETERS

    OFFLOAD_PARFOR_1D_PARAM(i, rllm::enum_iterator<rllm::PositionIndex>(static_cast<rllm::PositionIndex>(N)), (visits))
    ATOMIC_INC(visits[static_cast<size_t>(i)]);
    ENDFOR

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(visits[i], 1) << "Wrong visit count at i=" << i;
}

TEST_F(OffloadParForTest, OffloadParFor2DVisitsEachCellExactlyOnce)
{
    // OFFLOAD_PARAMETERS(visits,ROWS,COLS)
    constexpr size_t ROWS = 7;
    constexpr size_t COLS = 11;
    rllm::fixed_size_matrix<int, rllm::PositionIndex, rllm::HeadDimension> visits;
    // END_OFFLOAD_PARAMETERS

    const auto grid = rllm::enum_iterator2D<rllm::PositionIndex, rllm::HeadDimension>(
        static_cast<rllm::PositionIndex>(ROWS), static_cast<rllm::HeadDimension>(COLS)
    );
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    ATOMIC_INC((visits[i, j]));
    ENDFOR

    for (size_t i = 0; i < ROWS; ++i)
        for (size_t j = 0; j < COLS; ++j)
        {
            EXPECT_EQ((visits[static_cast<rllm::PositionIndex>(i), static_cast<rllm::HeadDimension>(j)]), 1)
                << "Wrong visit count at (" << i << "," << j << ")";
        }
}

TEST_F(OffloadParForTest, OffloadParFor2DParamVisitsEachCellExactlyOnce)
{
    // OFFLOAD_PARAMETERS(visits,ROWS,COLS)
    constexpr size_t ROWS = 5;
    constexpr size_t COLS = 13;
    rllm::fixed_size_matrix<int, rllm::PositionIndex, rllm::HeadDimension> visits;
    // END_OFFLOAD_PARAMETERS

    const auto rows = static_cast<rllm::PositionIndex>(ROWS);
    const auto cols = static_cast<rllm::HeadDimension>(COLS);

    const auto grid = rllm::enum_iterator2D<rllm::PositionIndex, rllm::HeadDimension>(rows, cols);
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    ATOMIC_INC((visits[i, j]));
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ);

    for (size_t i = 0; i < ROWS; ++i)
        for (size_t j = 0; j < COLS; ++j)
        {
            EXPECT_EQ((visits[static_cast<rllm::PositionIndex>(i), static_cast<rllm::HeadDimension>(j)]), 1)
                << "Wrong visit count at (" << i << "," << j << ")";
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ);
}

TEST_F(OffloadParForTest, OffloadParFor2DTriangularParamVisitsLowerTriangle)
{
    // OFFLOAD_PARAMETERS(visits,N)
    constexpr size_t N = 8;
    rllm::fixed_size_matrix<int, rllm::PositionIndex, rllm::PositionIndex> visits;
    // END_OFFLOAD_PARAMETERS

    OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, static_cast<rllm::PositionIndex>(N), (visits))
    ATOMIC_INC((visits[i, j]));
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ);

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            const int expected = (j <= i) ? 1 : 0;
            EXPECT_EQ((visits[static_cast<rllm::PositionIndex>(i), static_cast<rllm::PositionIndex>(j)]), expected)
                << "Wrong visit count at (" << i << "," << j << ")";
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ);
}

TEST_F(OffloadParForTest, OffloadParFor2DUpperTriangularParamVisitsUpperTriangle)
{
    // OFFLOAD_PARAMETERS(visits,N)
    constexpr size_t N = 8;
    rllm::fixed_size_matrix<int, rllm::PositionIndex, rllm::PositionIndex> visits;
    // END_OFFLOAD_PARAMETERS

    OFFLOAD_PARFOR_2D_UPPER_TRIANGULAR_PARAM(i, j, static_cast<rllm::PositionIndex>(N), (visits))
    ATOMIC_INC((visits[i, j]));
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ);

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            const int expected = (i <= j) ? 1 : 0;
            EXPECT_EQ((visits[static_cast<rllm::PositionIndex>(i), static_cast<rllm::PositionIndex>(j)]), expected)
                << "Wrong visit count at (" << i << "," << j << ")";
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ);
}


/** Test that between two OFFLOAD_PARFOR_2D_PARAM calls with the same parameters,
 *  the second call does not cause additional host-device buffer copies because the data is still valid on the device from the first call.
 */
TEST_F(OffloadParForTest, OffloadParFor2DParamVisitsEachCellTwiceInARow)
{
    // OFFLOAD_PARAMETERS(visits,ROWS,COLS)
    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 9;
    DevicePointer<int> visits(ROWS * COLS);
    // END_OFFLOAD_PARAMETERS
    for (size_t idx = 0; idx < visits.size(); ++idx)
        visits[idx] = 0;

    const auto grid = rllm::enum_iterator2D<rllm::PositionIndex, rllm::HeadDimension>(
        static_cast<rllm::PositionIndex>(ROWS), static_cast<rllm::HeadDimension>(COLS)
    );

    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    const size_t idx = static_cast<size_t>(i) * COLS + static_cast<size_t>(j);
    ATOMIC_INC(visits[idx]);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    const size_t idx = static_cast<size_t>(i) * COLS + static_cast<size_t>(j);
    ATOMIC_INC(visits[idx]);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (size_t i = 0; i < ROWS; ++i)
        for (size_t j = 0; j < COLS; ++j)
        {
            const size_t idx = i * COLS + j;
            EXPECT_EQ(visits[idx], 2) // by reading the DevicePointer we cause ONE device-to-host copy.
                << "Wrong visit count at (" << i << "," << j << ")";
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor1DParamVisitsEachIndexExactlyOnceUsingFixedSizeVector)
{
    // OFFLOAD_PARAMETERS(visits,N)
    constexpr size_t N = 21;
    rllm::fixed_size_vector<int, rllm::PositionIndex> visits;
    visits.set_size(static_cast<rllm::PositionIndex>(N));
    // END_OFFLOAD_PARAMETERS
    visits.fill(0, static_cast<rllm::PositionIndex>(N));

    OFFLOAD_PARFOR_1D_PARAM(i, rllm::enum_iterator<rllm::PositionIndex>(static_cast<rllm::PositionIndex>(N)), (visits))
    visits[static_cast<size_t>(i)] = static_cast<int>(i) + 42;
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(visits[i], static_cast<int>(i) + 42) << "Wrong value at i=" << i;

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor1DParamWritesFixedSizeFloatVector)
{
    // OFFLOAD_PARAMETERS(values,N)
    constexpr size_t N = 21;
    rllm::fixed_size_vector<rllm::rlmm_float, rllm::PositionIndex> values;
    values.set_size(static_cast<rllm::PositionIndex>(N));
    // END_OFFLOAD_PARAMETERS
    values.fill(0.0f, static_cast<rllm::PositionIndex>(N));

    OFFLOAD_PARFOR_1D_PARAM(i, rllm::enum_iterator<rllm::PositionIndex>(static_cast<rllm::PositionIndex>(N)), (values))
    values[static_cast<size_t>(i)] = static_cast<rllm::rlmm_float>(static_cast<float>(i) + 0.5f);
    ENDFOR

    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(values[i], static_cast<float>(i) + 0.5f) << "Wrong value at i=" << i;
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFixedSizeMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    rllm::fixed_size_matrix<float, rllm::MultiTokenPredictionIndex, rllm::HeadsIndex> values;
    // END_OFFLOAD_PARAMETERS
    values.fill(0.0f);

    const auto grid = rllm::enum_iterator2D<rllm::MultiTokenPredictionIndex, rllm::HeadsIndex>();
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[i, j] = static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 1);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (const auto i : rllm::enum_iterator<rllm::MultiTokenPredictionIndex>())
        for (const auto j : rllm::enum_iterator<rllm::HeadsIndex>())
            ASSERT_FLOAT_EQ(
                (values[i, j]),
                static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 1)
            );

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}


TEST_F(OffloadParForTest, InputLayerPropagateForwardMatchesReference)
{
    std::srand(0);

    rllm::InputLayer input_layer;
    input_layer.set_random_embeddings();

    rllm::InputLine input;
    input.push_back(static_cast<rllm::TokenID>(1));
    input.push_back(static_cast<rllm::TokenID>(7));
    input.push_back(static_cast<rllm::TokenID>(11));
    input.push_back(static_cast<rllm::TokenID>(19));

    rllm::flexible_rows_matrix<rllm::rlmm_float, rllm::PositionIndex, rllm::EmbeddingDimension> actual(input.size());
    input_layer.propagate_forward(input, actual);

    constexpr float model_dim = static_cast<float>(rllm::EmbeddingDimension::MAX);
    constexpr float max_abs_error = 1e-5f;

    for (const auto pos : rllm::enum_iterator<rllm::PositionIndex>(input.size()))
    {
        const auto embedding = input_layer.get_embedding(input[pos]);
        const float pos_f = static_cast<float>(pos);

        for (const auto di : rllm::enum_iterator<rllm::EmbeddingDimension>())
        {
            const int di_int = static_cast<int>(di);
            const float emb_val = embedding[di];
            const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(di_int & ~1) / model_dim);
            const float pe = (di_int % 2 == 0) ? std::sin(pos_f * freq) : std::cos(pos_f * freq);
            const float expected = emb_val + pe;
            const float actual_value = actual[pos, di];

            ASSERT_NEAR(actual_value, expected, max_abs_error)
                << "Mismatch at pos=" << static_cast<size_t>(pos)
                << ", di=" << static_cast<size_t>(di);
        }
    }
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFixedSizeMatrixUsingFloatParam)
{
    // OFFLOAD_PARAMETERS(values, scale)
    rllm::fixed_size_matrix<float, rllm::MultiTokenPredictionIndex, rllm::HeadsIndex> values;
    float scale = 0.25f;
    // END_OFFLOAD_PARAMETERS
    values.fill(0.0f);

    const auto grid = rllm::enum_iterator2D<rllm::MultiTokenPredictionIndex, rllm::HeadsIndex>();
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values, scale))
    values[i, j] = scale * static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 5);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (const auto i : rllm::enum_iterator<rllm::MultiTokenPredictionIndex>())
        for (const auto j : rllm::enum_iterator<rllm::HeadsIndex>())
            EXPECT_FLOAT_EQ(
                (values[i, j]),
                scale * static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 5)
            );

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor2DParamPerformanceComparedToParFor2D)
{
    constexpr int BENCHMARK_RUNS = 1000;

    // OFFLOAD_PARAMETERS(offload_values, scale, WORK_PER_CELL, ITERS)
    rllm::fixed_size_matrix<float, rllm::EmbeddingDimension, rllm::EmbeddingDimension> offload_values;
    float scale = 0.25f;
    const int WORK_PER_CELL = 64;
    const int ITERS = 40;
    // END_OFFLOAD_PARAMETERS
    rllm::fixed_size_matrix<float, rllm::EmbeddingDimension, rllm::EmbeddingDimension> parfor_values;

    const auto grid = rllm::enum_iterator2D<rllm::EmbeddingDimension, rllm::EmbeddingDimension>();

    auto run_parfor = [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        PARFOR_2D(j, i, grid)
        const float base = static_cast<float>(static_cast<size_t>(j) * 3 + static_cast<size_t>(i));
        float v = parfor_values[j, i];
        for (int it = 0; it < ITERS; ++it)
        {
            for (int k = 0; k < WORK_PER_CELL; ++k)
                v += scale * (base + static_cast<float>(k));
        }
        parfor_values[j, i] = v;
        ENDFOR
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    };

    auto run_offload = [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (offload_values, scale, WORK_PER_CELL, ITERS))
        const float base = static_cast<float>(static_cast<size_t>(j) * 3 + static_cast<size_t>(i));
        float v = offload_values[j, i];
        for (int it = 0; it < ITERS; ++it)
        {
            for (int k = 0; k < WORK_PER_CELL; ++k)
                v += scale * (base + static_cast<float>(k));
        }
        offload_values[j, i] = v;
        ENDFOR
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    };

    // Warm both paths once so the timed run compares steady-state execution.
    parfor_values.fill(0.0f);
    static_cast<void>(run_parfor());
    offload_values.fill(0.0f);
    static_cast<void>(run_offload());

    long long parfor_us = 0;
    long long offload_us = 0;
    for (int run = 0; run < BENCHMARK_RUNS; ++run)
    {
        if (run == BENCHMARK_RUNS - 1)
            parfor_values.fill(0.0f);
        parfor_us += run_parfor();
    }
    for (int run = 0; run < BENCHMARK_RUNS; ++run)
    {
        if (run == BENCHMARK_RUNS - 1)
        {
            offload_values.fill(0.0f);
            parallel::statistics.reset_buffer_copy_counters();
        }
        offload_us += run_offload();
    }
    const double speedup = static_cast<double>(parfor_us) / static_cast<double>(offload_us);

    RecordProperty("parfor_us", parfor_us);
    RecordProperty("offload_us", offload_us);
    RecordProperty("offload_vs_parfor_speedup", speedup);

    fprintf(
        stderr,
        "OFFLOAD_PARFOR_2D_PARAM vs PARFOR_2D (%zu x %zu, work=%d, iters=%d, runs=%d)"
        " - PARFOR: %lldus, OFFLOAD: %lldus, Speedup: %.2fx\n",
        static_cast<size_t>(rllm::EmbeddingDimension::MAX),
        static_cast<size_t>(rllm::EmbeddingDimension::MAX),
        WORK_PER_CELL,
        ITERS,
        BENCHMARK_RUNS,
        static_cast<long long>(parfor_us),
        static_cast<long long>(offload_us),
        speedup
    );

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (const auto i : rllm::enum_iterator<rllm::EmbeddingDimension>())
        for (const auto j : rllm::enum_iterator<rllm::EmbeddingDimension>())
            EXPECT_FLOAT_EQ(
                (offload_values[i, j]),
                (parfor_values[i, j])
            );

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFlexibleRowsMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    rllm::flexible_rows_matrix<float, rllm::MultiTokenPredictionIndex, rllm::HeadsIndex> values(static_cast<rllm::MultiTokenPredictionIndex>(3));
    // END_OFFLOAD_PARAMETERS
    values.fill(0.0f);

    const auto grid = rllm::enum_iterator2D<rllm::MultiTokenPredictionIndex, rllm::HeadsIndex>(
        static_cast<rllm::MultiTokenPredictionIndex>(3)
    );
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[i, j] = static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 2);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (const auto i : rllm::enum_iterator<rllm::MultiTokenPredictionIndex>(rllm::MultiTokenPredictionIndex::THREE))
        for (const auto j : rllm::enum_iterator<rllm::HeadsIndex>())
        {
            const float actual_value = values[i, j];
            EXPECT_FLOAT_EQ(
                actual_value,
                static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 2)
            );
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFlexibleColsMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    rllm::flexible_cols_matrix<float, rllm::MultiTokenPredictionIndex, rllm::HeadsIndex> values(rllm::HeadsIndex::MAX);
    // END_OFFLOAD_PARAMETERS
    values.fill(0.0f);

    const auto grid = rllm::enum_iterator2D<rllm::MultiTokenPredictionIndex, rllm::HeadsIndex>();
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[i, j] = static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 3);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (const auto i : rllm::enum_iterator<rllm::MultiTokenPredictionIndex>(static_cast<rllm::MultiTokenPredictionIndex>(3)))
        for (const auto j : rllm::enum_iterator<rllm::HeadsIndex>())
        {
            const float actual_value = values[i, j];
            EXPECT_FLOAT_EQ(
                actual_value,
                static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 3)
            );
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFlexibleRowsColsMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    rllm::flexible_rows_cols_matrix<float, rllm::MultiTokenPredictionIndex, rllm::HeadsIndex> values(static_cast<rllm::MultiTokenPredictionIndex>(3), rllm::HeadsIndex::MAX);
    // END_OFFLOAD_PARAMETERS
    values.fill(0.0f);

    const auto grid = rllm::enum_iterator2D<rllm::MultiTokenPredictionIndex, rllm::HeadsIndex>(
        static_cast<rllm::MultiTokenPredictionIndex>(3)
    );
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[i, j] = static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 4);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (const auto i : rllm::enum_iterator<rllm::MultiTokenPredictionIndex>(static_cast<rllm::MultiTokenPredictionIndex>(3)))
        for (const auto j : rllm::enum_iterator<rllm::HeadsIndex>())
        {
            const float actual_value = values[i, j];
            EXPECT_FLOAT_EQ(
                actual_value,
                static_cast<float>(static_cast<size_t>(i) * 100 + static_cast<size_t>(j) + 4)
            );
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

#endif

#include <gtest/gtest.h>

#include <InputLayer.hpp>
#include <LayerPrimitives.hpp>
#include <device_pointer.hpp>
#include <enum_iterator1D.hpp>
#include <enum_iterator2D.hpp>
#include <fixed_size_matrix.hpp>
#include <fixed_size_vector.hpp>
#include <parallel.hpp>
#include <vecmath.hpp>
#include <cpu/cpu_fixed_vector.hpp>
#include <cpu/cpu_fixed_matrix.hpp>
#include <cpu/cpu_flex_rows_matrix.hpp>
#include <cpu/cpu_flex_cols_matrix.hpp>
#include <cpu/cpu_flex_rows_cols_matrix.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace rlmm;

#if defined(USE_VULKAN_OFFLOAD)
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES = 0u;
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ = 0u;
constexpr size_t EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ = 1u;
#else
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

static void visit_lower_triangle(
    // OFFLOAD_PARAMETERS(visits, n)
    fixed_size_matrix<int, PositionIndex, PositionIndex>& visits,
    PositionIndex n
    // END_OFFLOAD_PARAMETERS
)
{
    OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, n, (visits))
    atomicAdd((visits[i, j]), 1);
    ENDFOR
}

static void visit_upper_triangle(
    // OFFLOAD_PARAMETERS(visits, n)
    fixed_size_matrix<int, PositionIndex, PositionIndex>& visits,
    PositionIndex n
    // END_OFFLOAD_PARAMETERS
)
{
    OFFLOAD_PARFOR_2D_UPPER_TRIANGULAR_PARAM(i, j, n, (visits))
    atomicAdd((visits[i, j]), 1);
    ENDFOR
}

TEST_F(OffloadParForTest, OffloadParForVisitsEachIndexExactlyOnce)
{
    constexpr size_t N = 17;
    // OFFLOAD_PARAMETERS(visits)
    fixed_size_vector<int, PositionIndex> visits;
    // END_OFFLOAD_PARAMETERS

    visits.set_size(static_cast<PositionIndex>(N));
    
    OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<PositionIndex>(static_cast<PositionIndex>(N)), (visits))
    atomicAdd(visits[static_cast<size_t>(i)], 1);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ);

    cpu_fixed_vector<int, PositionIndex> cpu_visits;
    visits.copy_to_cpu(cpu_visits);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(cpu_visits[i], 1) << "Wrong visit count at i=" << i;

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ);
}

TEST_F(OffloadParForTest, OffloadParForParamVisitsEachIndexExactlyOnce)
{
    constexpr size_t N = 19;
    // PARFOR_PARAM requires the loop variable to be captured in a lambda, so we can't use a simple array of ints here.
    // OFFLOAD_PARAMETERS(visits)
    fixed_size_vector<int, PositionIndex> visits;
    // END_OFFLOAD_PARAMETERS

    visits.set_size(static_cast<PositionIndex>(N));

    OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<PositionIndex>(static_cast<PositionIndex>(N)), (visits))
    atomicAdd(visits[static_cast<size_t>(i)], 1);
    ENDFOR

    cpu_fixed_vector<int, PositionIndex> cpu_visits2;
    visits.copy_to_cpu(cpu_visits2);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(cpu_visits2[i], 1) << "Wrong visit count at i=" << i;
}

TEST_F(OffloadParForTest, OffloadParFor2DVisitsEachCellExactlyOnce)
{
    // OFFLOAD_PARAMETERS(visits,ROWS,COLS)
    constexpr size_t ROWS = 7;
    constexpr size_t COLS = 11;
    fixed_size_matrix<int, PositionIndex, HeadDimension> visits;
    // END_OFFLOAD_PARAMETERS

    const auto grid = enum_iterator2D<PositionIndex, HeadDimension>(
        static_cast<PositionIndex>(ROWS), static_cast<HeadDimension>(COLS)
    );
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    atomicAdd((visits[i, j]), 1);
    ENDFOR

    cpu_fixed_matrix<int, PositionIndex, HeadDimension> cpu_visits3;
    visits.copy_to_cpu(cpu_visits3);
    for (size_t i = 0; i < ROWS; ++i)
        for (size_t j = 0; j < COLS; ++j)
        {
            EXPECT_EQ((cpu_visits3[static_cast<PositionIndex>(i), static_cast<HeadDimension>(j)]), 1)
                << "Wrong visit count at (" << i << "," << j << ")";
        }
}

TEST_F(OffloadParForTest, OffloadParFor2DParamVisitsEachCellExactlyOnce)
{
    // OFFLOAD_PARAMETERS(visits,ROWS,COLS)
    constexpr size_t ROWS = 5;
    constexpr size_t COLS = 13;
    fixed_size_matrix<int, PositionIndex, HeadDimension> visits;
    // END_OFFLOAD_PARAMETERS

    const auto rows = static_cast<PositionIndex>(ROWS);
    const auto cols = static_cast<HeadDimension>(COLS);

    const auto grid = enum_iterator2D<PositionIndex, HeadDimension>(rows, cols);
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    atomicAdd((visits[i, j]), 1);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ);

    cpu_fixed_matrix<int, PositionIndex, HeadDimension> cpu_visits4;
    visits.copy_to_cpu(cpu_visits4);
    for (size_t i = 0; i < ROWS; ++i)
        for (size_t j = 0; j < COLS; ++j)
        {
            EXPECT_EQ((cpu_visits4[static_cast<PositionIndex>(i), static_cast<HeadDimension>(j)]), 1)
                << "Wrong visit count at (" << i << "," << j << ")";
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ);
}

TEST_F(OffloadParForTest, OffloadParFor2DTriangularParamVisitsLowerTriangle)
{
    // OFFLOAD_PARAMETERS(visits,N)
    constexpr size_t N = 8;
    fixed_size_matrix<int, PositionIndex, PositionIndex> visits;
    // END_OFFLOAD_PARAMETERS

    visit_lower_triangle(visits, static_cast<PositionIndex>(N));

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ);

    cpu_fixed_matrix<int, PositionIndex, PositionIndex> cpu_tri_lower;
    visits.copy_to_cpu(cpu_tri_lower);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            const int expected = (j <= i) ? 1 : 0;
            EXPECT_EQ((cpu_tri_lower[static_cast<PositionIndex>(i), static_cast<PositionIndex>(j)]), expected)
                << "Wrong visit count at (" << i << "," << j << ")";
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_AFTER_READ);
}

TEST_F(OffloadParForTest, OffloadParFor2DUpperTriangularParamVisitsUpperTriangle)
{
    // OFFLOAD_PARAMETERS(visits,N)
    constexpr size_t N = 8;
    fixed_size_matrix<int, PositionIndex, PositionIndex> visits;
    // END_OFFLOAD_PARAMETERS

    visit_upper_triangle(visits, static_cast<PositionIndex>(N));

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_D2H_COPIES_BEFORE_READ);

    cpu_fixed_matrix<int, PositionIndex, PositionIndex> cpu_tri_upper;
    visits.copy_to_cpu(cpu_tri_upper);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            const int expected = (i <= j) ? 1 : 0;
            EXPECT_EQ((cpu_tri_upper[static_cast<PositionIndex>(i), static_cast<PositionIndex>(j)]), expected)
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
    fixed_size_vector<int, PositionIndex> visits;
    // END_OFFLOAD_PARAMETERS
    visits.set_size(static_cast<PositionIndex>(ROWS * COLS));

    visits.zero();

    const auto grid = enum_iterator2D<PositionIndex, HeadDimension>(
        static_cast<PositionIndex>(ROWS), static_cast<HeadDimension>(COLS)
    );

    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    const size_t idx = ((static_cast<size_t>(i) * COLS) + static_cast<size_t>(j));
    atomicAdd(visits[idx], 1);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 0u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    const size_t idx = ((static_cast<size_t>(i) * COLS) + static_cast<size_t>(j));
    atomicAdd(visits[idx], 1);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 0u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    cpu_fixed_vector<int, PositionIndex> cpu_twice;
    visits.copy_to_cpu(cpu_twice);
    for (size_t i = 0; i < ROWS; ++i)
        for (size_t j = 0; j < COLS; ++j)
        {
            const size_t idx = i * COLS + j;
            EXPECT_EQ(cpu_twice[idx], 2)
                << "Wrong visit count at (" << i << "," << j << ")";
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor1DParamVisitsEachIndexExactlyOnceUsingFixedSizeVector)
{
    // OFFLOAD_PARAMETERS(visits,N)
    constexpr size_t N = 21;
    fixed_size_vector<int, PositionIndex> visits;
    // END_OFFLOAD_PARAMETERS

    visits.set_size(static_cast<PositionIndex>(N));
    fill(visits, 0, static_cast<PositionIndex>(N));

    OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<PositionIndex>(static_cast<PositionIndex>(N)), (visits))
    visits[static_cast<size_t>(i)] = (static_cast<int>(i) + 42);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    cpu_fixed_vector<int, PositionIndex> cpu_int_vec;
    visits.copy_to_cpu(cpu_int_vec);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(cpu_int_vec[i], (static_cast<int>(i) + 42)) << "Wrong value at i=" << i;

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor1DParamWritesFixedSizeFloatVector)
{
    // OFFLOAD_PARAMETERS(values,N)
    constexpr size_t N = 21;
    fixed_size_vector<float, PositionIndex> values;
    // END_OFFLOAD_PARAMETERS
    values.set_size(static_cast<PositionIndex>(N));
    fill(values, 0.0f, static_cast<PositionIndex>(N));

    OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<PositionIndex>(static_cast<PositionIndex>(N)), (values))
    values[static_cast<size_t>(i)] = static_cast<float>((static_cast<float>(i) + 0.5f));
    ENDFOR

    cpu_fixed_vector<float, PositionIndex> cpu_float_vec;
    values.copy_to_cpu(cpu_float_vec);
    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(cpu_float_vec[i], (static_cast<float>(i) + 0.5f)) << "Wrong value at i=" << i;
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFixedSizeMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    fixed_size_matrix<float, MultiTokenPredictionIndex, HeadsIndex> values;
    // END_OFFLOAD_PARAMETERS
    values.zero();

    const auto grid = enum_iterator2D<MultiTokenPredictionIndex, HeadsIndex>();
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[i, j] = static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 1)));
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    cpu_fixed_matrix<float, MultiTokenPredictionIndex, HeadsIndex> cpu_mat1;
    values.copy_to_cpu(cpu_mat1);
    for (const auto i : enum_iterator1D<MultiTokenPredictionIndex>())
        for (const auto j : enum_iterator1D<HeadsIndex>())
            ASSERT_FLOAT_EQ(
                (cpu_mat1[i, j]),
                static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 1)))
            );

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}


TEST_F(OffloadParForTest, InputLayerPropagateForwardMatchesReference)
{
    std::srand(0);

    InputLayer input_layer;
    input_layer.set_random_embeddings();

    InputLine input;
    input.push_back(static_cast<TokenID>(1));
    input.push_back(static_cast<TokenID>(7));
    input.push_back(static_cast<TokenID>(11));
    input.push_back(static_cast<TokenID>(19));

    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> actual(input.size());
    input_layer.propagate_forward(input, actual);

    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> cpu_actual;
    actual.copy_to_cpu(cpu_actual);

    constexpr float model_dim = static_cast<float>(EmbeddingDimension::MAX);
    constexpr float max_abs_error = 1e-5f;

    for (const auto pos : enum_iterator1D<PositionIndex>(input.size()))
    {
        embedding_row_t embedding;
        input_layer.get_embedding(input[pos], embedding);
        const float pos_f = static_cast<float>(pos);

        for (const auto di : enum_iterator1D<EmbeddingDimension>())
        {
            const int di_int = static_cast<int>(di);
            const float emb_val = embedding[static_cast<size_t>(di)];
            const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(di_int & ~1) / model_dim);
            const float pe = (di_int % 2 == 0) ? std::sin(pos_f * freq) : std::cos(pos_f * freq);
            const float expected = emb_val + pe;
            const float actual_value = cpu_actual[pos, di];

            ASSERT_NEAR(actual_value, expected, max_abs_error)
                << "Mismatch at pos=" << static_cast<size_t>(pos)
                << ", di=" << static_cast<size_t>(di);
        }
    }
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFixedSizeMatrixUsingFloatParam)
{
    // OFFLOAD_PARAMETERS(values, scale)
    fixed_size_matrix<float, MultiTokenPredictionIndex, HeadsIndex> values;
    float scale = 0.25f;
    // END_OFFLOAD_PARAMETERS
    values.zero();

    const auto grid = enum_iterator2D<MultiTokenPredictionIndex, HeadsIndex>();
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values, scale))
    values[i, j] = (scale * static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 5))));
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    cpu_fixed_matrix<float, MultiTokenPredictionIndex, HeadsIndex> cpu_mat2;
    values.copy_to_cpu(cpu_mat2);
    for (const auto i : enum_iterator1D<MultiTokenPredictionIndex>())
        for (const auto j : enum_iterator1D<HeadsIndex>())
            EXPECT_FLOAT_EQ(
                (cpu_mat2[i, j]),
                (scale * static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 5))))
            );

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor2DParamPerformanceComparedToParFor2D)
{
    constexpr int BENCHMARK_RUNS = 1000;

    // OFFLOAD_PARAMETERS(offload_values, scale, WORK_PER_CELL, ITERS)
    fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> offload_values;
    float scale = 0.25f;
    const int WORK_PER_CELL = 64;
    const int ITERS = 40;
    // END_OFFLOAD_PARAMETERS
    cpu_fixed_matrix<float, EmbeddingDimension, EmbeddingDimension> parfor_values;

    const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();

    auto run_parfor = [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        PARFOR_2D(j, i, grid)
        const float base = static_cast<float>(((static_cast<size_t>(j) * 3) + static_cast<size_t>(i)));
        float v = parfor_values[j, i];
        for (int it = 0; it < ITERS; ++it)
        {
            for (int k = 0; k < WORK_PER_CELL; ++k)
                v += (scale * (base + static_cast<float>(k)));
        }
        parfor_values[j, i] = v;
        ENDFOR
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    };

    auto run_offload = [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (offload_values, scale, WORK_PER_CELL, ITERS))
        const float base = static_cast<float>(((static_cast<size_t>(j) * 3) + static_cast<size_t>(i)));
        float v = offload_values[j, i];
        for (int it = 0; it < ITERS; ++it)
        {
            for (int k = 0; k < WORK_PER_CELL; ++k)
                v += (scale * (base + static_cast<float>(k)));
        }
        offload_values[j, i] = v;
        ENDFOR
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    };

    // Warm both paths once so the timed run compares steady-state execution.
    parfor_values.zero();
    static_cast<void>(run_parfor());
    offload_values.zero();
    static_cast<void>(run_offload());

    long long parfor_us = 0;
    long long offload_us = 0;
    for (int run = 0; run < BENCHMARK_RUNS; ++run)
    {
        if (run == BENCHMARK_RUNS - 1)
            parfor_values.zero();
        parfor_us += run_parfor();
    }
    for (int run = 0; run < BENCHMARK_RUNS; ++run)
    {
        if (run == BENCHMARK_RUNS - 1)
        {
            offload_values.zero();
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
        static_cast<size_t>(EmbeddingDimension::MAX),
        static_cast<size_t>(EmbeddingDimension::MAX),
        WORK_PER_CELL,
        ITERS,
        BENCHMARK_RUNS,
        static_cast<long long>(parfor_us),
        static_cast<long long>(offload_us),
        speedup
    );

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    cpu_fixed_matrix<float, EmbeddingDimension, EmbeddingDimension> cpu_offload;
    offload_values.copy_to_cpu(cpu_offload);
    for (const auto i : enum_iterator1D<EmbeddingDimension>())
        for (const auto j : enum_iterator1D<EmbeddingDimension>())
            EXPECT_FLOAT_EQ(
                (cpu_offload[i, j]),
                (parfor_values[i, j])
            );

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFlexibleRowsMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    flexible_rows_matrix<float, MultiTokenPredictionIndex, HeadsIndex> values;
    // END_OFFLOAD_PARAMETERS
    values.set_rows(static_cast<MultiTokenPredictionIndex>(3));
    values.zero();

    const auto grid = enum_iterator2D<MultiTokenPredictionIndex, HeadsIndex>(
        static_cast<MultiTokenPredictionIndex>(3)
    );
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[i, j] = static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 2)));
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    cpu_flex_rows_matrix<float, MultiTokenPredictionIndex, HeadsIndex> cpu_flex_rows;
    values.copy_to_cpu(cpu_flex_rows);
    for (const auto i : enum_iterator1D<MultiTokenPredictionIndex>(MultiTokenPredictionIndex::THREE))
        for (const auto j : enum_iterator1D<HeadsIndex>())
        {
            const float actual_value = cpu_flex_rows[i, j];
            EXPECT_FLOAT_EQ(
                actual_value,
                static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 2)))
            );
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFlexibleColsMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    flexible_cols_matrix<float, MultiTokenPredictionIndex, HeadsIndex> values;
    // END_OFFLOAD_PARAMETERS
    values.set_cols(HeadsIndex::MAX);
    values.zero();

    const auto grid = enum_iterator2D<MultiTokenPredictionIndex, HeadsIndex>();
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[i, j] = static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 3)));
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    cpu_flex_cols_matrix<float, MultiTokenPredictionIndex, HeadsIndex> cpu_flex_cols;
    values.copy_to_cpu(cpu_flex_cols);
    for (const auto i : enum_iterator1D<MultiTokenPredictionIndex>(static_cast<MultiTokenPredictionIndex>(3)))
        for (const auto j : enum_iterator1D<HeadsIndex>())
        {
            const float actual_value = cpu_flex_cols[i, j];
            EXPECT_FLOAT_EQ(
                actual_value,
                static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 3)))
            );
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFlexibleRowsColsMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    flexible_rows_cols_matrix<float, MultiTokenPredictionIndex, HeadsIndex> values;
    // END_OFFLOAD_PARAMETERS
    values.set_size(static_cast<MultiTokenPredictionIndex>(3), HeadsIndex::MAX);
    values.zero();

    const auto grid = enum_iterator2D<MultiTokenPredictionIndex, HeadsIndex>(
        static_cast<MultiTokenPredictionIndex>(3)
    );
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[i, j] = static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 4)));
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), EXPECTED_FIXED_SIZE_OFFLOAD_H2D_COPIES);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    cpu_flex_rows_cols_matrix<float, MultiTokenPredictionIndex, HeadsIndex> cpu_flex_rc;
    values.copy_to_cpu(cpu_flex_rc);
    for (const auto i : enum_iterator1D<MultiTokenPredictionIndex>(static_cast<MultiTokenPredictionIndex>(3)))
        for (const auto j : enum_iterator1D<HeadsIndex>())
        {
            const float actual_value = cpu_flex_rc[i, j];
            EXPECT_FLOAT_EQ(
                actual_value,
                static_cast<float>(((static_cast<size_t>(i) * 100) + (static_cast<size_t>(j) + 4)))
            );
        }

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);
}

#endif

#include <gtest/gtest.h>

#include <LayerPrimitives.hpp>
#include <device_pointer.hpp>
#include <enum_iterator.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <vector>

#if defined(USE_VULKAN_OFFLOAD)
#define ATOMIC_INC(x) ((x)++)
#else
#define ATOMIC_INC(x) ((x).fetch_add(1, std::memory_order_relaxed))
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
    std::vector<std::atomic<int>> visits(N);
    // END_OFFLOAD_PARAMETERS
    for (auto& v : visits)
        v.store(0, std::memory_order_relaxed);

    OFFLOAD_PARFOR_1D_PARAM(i, rllm::enum_iterator<rllm::PositionIndex>(static_cast<rllm::PositionIndex>(N)), (visits))
    ATOMIC_INC(visits[static_cast<size_t>(i)]);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(visits[i].load(std::memory_order_relaxed), 1) << "Wrong visit count at i=" << i;
}

TEST_F(OffloadParForTest, OffloadParForParamVisitsEachIndexExactlyOnce)
{
    constexpr size_t N = 19;
    // PARFOR_PARAM requires the loop variable to be captured in a lambda, so we can't use a simple array of ints here.
    // Instead, we use atomics to track visits.

    // OFFLOAD_PARAMETERS(visits)
    std::vector<std::atomic<int>> visits(N);
    // END_OFFLOAD_PARAMETERS

    // PARFOR_PARAM may execute iterations in any order and potentially in parallel, so we initialize the visit counts
    // to 0 before the loop.
    for (auto& v : visits)
        v.store(0, std::memory_order_relaxed);

    OFFLOAD_PARFOR_1D_PARAM(i, rllm::enum_iterator<rllm::PositionIndex>(static_cast<rllm::PositionIndex>(N)), (visits))
    ATOMIC_INC(visits[static_cast<size_t>(i)]);
    ENDFOR

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(visits[i].load(std::memory_order_relaxed), 1) << "Wrong visit count at i=" << i;
}

TEST_F(OffloadParForTest, OffloadParFor2DVisitsEachCellExactlyOnce)
{
    // OFFLOAD_PARAMETERS(visits,ROWS,COLS)
    constexpr size_t ROWS = 7;
    constexpr size_t COLS = 11;
    std::vector<std::atomic<int>> visits(ROWS * COLS);
    // END_OFFLOAD_PARAMETERS
    for (auto& v : visits)
        v.store(0, std::memory_order_relaxed);

    const auto grid = rllm::enum_iterator2D<rllm::PositionIndex, rllm::HeadDimension>(
        static_cast<rllm::PositionIndex>(ROWS), static_cast<rllm::HeadDimension>(COLS)
    );
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    const size_t idx = static_cast<size_t>(j) * COLS + static_cast<size_t>(i);
    ATOMIC_INC(visits[idx]);
    ENDFOR

    for (size_t i = 0; i < ROWS; ++i)
        for (size_t j = 0; j < COLS; ++j)
        {
            const size_t idx = i * COLS + j;
            EXPECT_EQ(visits[idx].load(std::memory_order_relaxed), 1)
                << "Wrong visit count at (" << i << "," << j << ")";
        }
}

TEST_F(OffloadParForTest, OffloadParFor2DParamVisitsEachCellExactlyOnce)
{
    // OFFLOAD_PARAMETERS(visits,ROWS,COLS)
    constexpr size_t ROWS = 5;
    constexpr size_t COLS = 13;
    std::vector<std::atomic<int>> visits(ROWS * COLS);
    // END_OFFLOAD_PARAMETERS

    const auto rows = static_cast<rllm::PositionIndex>(ROWS);
    const auto cols = static_cast<rllm::HeadDimension>(COLS);
    for (auto& v : visits)
        v.store(0, std::memory_order_relaxed);

    const auto grid = rllm::enum_iterator2D<rllm::PositionIndex, rllm::HeadDimension>(rows, cols);
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    const size_t idx = static_cast<size_t>(j) * COLS + static_cast<size_t>(i);
    ATOMIC_INC(visits[idx]);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);

    for (size_t i = 0; i < ROWS; ++i)
        for (size_t j = 0; j < COLS; ++j)
        {
            const size_t idx = i * COLS + j;
            EXPECT_EQ(visits[idx].load(std::memory_order_relaxed), 1)
                << "Wrong visit count at (" << i << "," << j << ")";
            }
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
    const size_t idx = static_cast<size_t>(j) * COLS + static_cast<size_t>(i);
    ATOMIC_INC(visits[idx]);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (visits))
    const size_t idx = static_cast<size_t>(j) * COLS + static_cast<size_t>(i);
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
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(visits[i], static_cast<int>(i) + 42) << "Wrong value at i=" << i;
}

TEST_F(OffloadParForTest, OffloadParFor2DParamWritesFixedSizeMatrix)
{
    // OFFLOAD_PARAMETERS(values)
    rllm::fixed_size_matrix<float, rllm::MultiTokenPredictionIndex, rllm::HeadsIndex> values;
    // END_OFFLOAD_PARAMETERS
    values.fill(0.0f);

    const auto grid = rllm::enum_iterator2D<rllm::MultiTokenPredictionIndex, rllm::HeadsIndex>();
    OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (values))
    values[j, i] = static_cast<float>(static_cast<size_t>(j) * 100 + static_cast<size_t>(i) + 1);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (size_t i = 0; i < static_cast<size_t>(rllm::MultiTokenPredictionIndex::MAX); ++i)
        for (size_t j = 0; j < static_cast<size_t>(rllm::HeadsIndex::MAX); ++j)
            ASSERT_FLOAT_EQ(
                (values[static_cast<rllm::MultiTokenPredictionIndex>(i), static_cast<rllm::HeadsIndex>(j)]),
                static_cast<float>(i * 100 + j + 1)
            );

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
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
    values[j, i] = scale * static_cast<float>(static_cast<size_t>(j) * 100 + static_cast<size_t>(i) + 5);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (size_t i = 0; i < static_cast<size_t>(rllm::MultiTokenPredictionIndex::MAX); ++i)
        for (size_t j = 0; j < static_cast<size_t>(rllm::HeadsIndex::MAX); ++j)
            EXPECT_FLOAT_EQ(
                (values[static_cast<rllm::MultiTokenPredictionIndex>(i), static_cast<rllm::HeadsIndex>(j)]),
                scale * static_cast<float>(i * 100 + j + 5)
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

    for (size_t i = 0; i < static_cast<size_t>(rllm::EmbeddingDimension::MAX); ++i)
        for (size_t j = 0; j < static_cast<size_t>(rllm::EmbeddingDimension::MAX); ++j)
            EXPECT_FLOAT_EQ(
                (offload_values[static_cast<rllm::EmbeddingDimension>(i), static_cast<rllm::EmbeddingDimension>(j)]),
                (parfor_values[static_cast<rllm::EmbeddingDimension>(i), static_cast<rllm::EmbeddingDimension>(j)])
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
    values[j, i] = static_cast<float>(static_cast<size_t>(j) * 100 + static_cast<size_t>(i) + 2);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < static_cast<size_t>(rllm::HeadsIndex::MAX); ++j)
            EXPECT_FLOAT_EQ(
                (values[static_cast<rllm::MultiTokenPredictionIndex>(i), static_cast<rllm::HeadsIndex>(j)]),
                static_cast<float>(i * 100 + j + 2)
            );

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
    values[j, i] = static_cast<float>(static_cast<size_t>(j) * 100 + static_cast<size_t>(i) + 3);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (size_t i = 0; i < static_cast<size_t>(rllm::MultiTokenPredictionIndex::MAX); ++i)
        for (size_t j = 0; j < static_cast<size_t>(rllm::HeadsIndex::MAX); ++j)
            EXPECT_FLOAT_EQ(
                (values[static_cast<rllm::MultiTokenPredictionIndex>(i), static_cast<rllm::HeadsIndex>(j)]),
                static_cast<float>(i * 100 + j + 3)
            );

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
    values[j, i] = static_cast<float>(static_cast<size_t>(j) * 100 + static_cast<size_t>(i) + 4);
    ENDFOR

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < static_cast<size_t>(rllm::HeadsIndex::MAX); ++j)
            EXPECT_FLOAT_EQ(
                (values[static_cast<rllm::MultiTokenPredictionIndex>(i), static_cast<rllm::HeadsIndex>(j)]),
                static_cast<float>(i * 100 + j + 4)
            );

    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);
}

#endif

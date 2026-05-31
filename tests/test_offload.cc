#include <gtest/gtest.h>

#include <LayerPrimitives.hpp>
#include <device_pointer.hpp>
#include <enum_iterator.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>

#include <atomic>
#include <vector>

#if defined(USE_VULKAN_OFFLOAD)
#define ATOMIC_INC(x) ((x)++)
#else
#define ATOMIC_INC(x) ((x).fetch_add(1, std::memory_order_relaxed))
#endif

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

#if defined(USE_OPENMP)
    GTEST_SKIP() << "OFFLOAD_PARFOR_PARAM is not defined in the OpenMP backend header.";
#else
    OFFLOAD_PARFOR_PARAM(i, rllm::enum_iterator<rllm::PositionIndex>(static_cast<rllm::PositionIndex>(N)), (visits))
    ATOMIC_INC(visits[static_cast<size_t>(i)]);
    ENDFOR
#endif

    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 1u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 1u);

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(visits[i].load(std::memory_order_relaxed), 1) << "Wrong visit count at i=" << i;
}

TEST_F(OffloadParForTest, OffloadParForParamVisitsEachIndexExactlyOnce)
{
#if defined(USE_OPENMP)
    GTEST_SKIP() << "OFFLOAD_PARFOR_PARAM is not defined in the OpenMP backend header.";
#else
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

    OFFLOAD_PARFOR_PARAM(i, rllm::enum_iterator<rllm::PositionIndex>(static_cast<rllm::PositionIndex>(N)), (visits))
    ATOMIC_INC(visits[static_cast<size_t>(i)]);
    ENDFOR

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(visits[i].load(std::memory_order_relaxed), 1) << "Wrong visit count at i=" << i;
#endif
}

TEST_F(OffloadParForTest, OffloadParFor2DVisitsEachCellExactlyOnce)
{
#if defined(USE_OPENMP)
    GTEST_SKIP() << "OFFLOAD_PARFOR_2D in this backend expands to a form that does not accept 2D range iterators.";
#else

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
#endif
}

TEST_F(OffloadParForTest, OffloadParFor2DParamVisitsEachCellExactlyOnce)
{
#if defined(USE_OPENMP)
    EXPECT_EQ(parallel::statistics.host_to_device_buffer_copies(), 0u);
    EXPECT_EQ(parallel::statistics.device_to_host_buffer_copies(), 0u);
    GTEST_SKIP() << "OFFLOAD_PARFOR_2D_PARAM uses an incompatible OpenMP macro shape in this configuration.";
#else
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
#endif
}


/** Test that between two OFFLOAD_PARFOR_2D_PARAM calls with the same parameters,
 *  the second call does not cause additional host-device buffer copies because the data is still valid on the device from the first call.
 */
TEST_F(OffloadParForTest, OffloadParFor2DParamVisitsEachCellTwiceInARow)
{
#if defined(USE_OPENMP)
    GTEST_SKIP() << "OFFLOAD_PARFOR_2D_PARAM uses an incompatible OpenMP macro shape in this configuration.";
#else
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
#endif
}

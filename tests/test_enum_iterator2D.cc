#include <gtest/gtest.h>

#include <LayerPrimitives.hpp>
#include <enum_iterator2D.hpp>

#include <vector>

// Visits every (Enum1, Enum2) pair exactly once (full 8x64 = 512 pairs).
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
            EXPECT_EQ(pairs[idx].first, h) << "wrong outer index at flat index " << idx;
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

// block_range should iterate a rectangular tile in strict row-major order.
TEST(EnumIterator2DTest, BlockRangeRowMajorAndBounds)
{
    using namespace rllm;

    const auto it = enum_iterator2D<HeadsIndex, HeadDimension>();

    // Rectangle: outer in [2,4), inner in [3,6)
    // Expected pairs:
    // (2,3) (2,4) (2,5) (3,3) (3,4) (3,5)
    std::vector<std::pair<int, int>> got;
    for (const auto [h, d] : it.block_range(/*outer_begin=*/2, /*outer_end=*/4, /*inner_begin=*/3, /*inner_end=*/6))
        got.emplace_back(static_cast<int>(h), static_cast<int>(d));

    const std::vector<std::pair<int, int>> expected = {
        {2, 3}, {2, 4}, {2, 5},
        {3, 3}, {3, 4}, {3, 5},
    };

    EXPECT_EQ(got, expected);
}

#include <gtest/gtest.h>

#include <LayerPrimitives.hpp>
#include <enum_iterator1D.hpp>

#include <vector>

TEST(EnumIteratorTest, TotalCount)
{
    using namespace rllm;
    int count = 0;
    for (const auto h : enum_iterator1D<HeadsIndex>())
        ++count;
    EXPECT_EQ(count, static_cast<int>(HeadsIndex::MAX));
}

TEST(EnumIteratorTest, DefaultStartsAtStartAndIsIncreasing)
{
    using namespace rllm;
    std::vector<int> values;
    values.reserve(static_cast<int>(HeadsIndex::MAX));

    for (const auto h : enum_iterator1D<HeadsIndex>())
        values.push_back(static_cast<int>(h));

    ASSERT_EQ(values.size(), static_cast<size_t>(static_cast<int>(HeadsIndex::MAX)));
    for (int i = 0; i < static_cast<int>(HeadsIndex::MAX); ++i)
        EXPECT_EQ(values[static_cast<size_t>(i)], i);
}

TEST(EnumIteratorTest, RuntimeUpperBound)
{
    using namespace rllm;
    constexpr int bound = 3;
    int count = 0;
    for (const auto h : enum_iterator1D<HeadsIndex>(static_cast<HeadsIndex>(bound)))
        ++count;
    EXPECT_EQ(count, bound);
}

TEST(EnumIteratorTest, StartEndRangeHonorsStart)
{
    using namespace rllm;
    std::vector<int> values;

    const auto start = static_cast<HeadsIndex>(2);
    const auto end = static_cast<HeadsIndex>(5);
    for (const auto h : enum_iterator1D<HeadsIndex>(start, end))
        values.push_back(static_cast<int>(h));

    const std::vector<int> expected = {2, 3, 4};
    EXPECT_EQ(values, expected);
}

TEST(EnumIteratorTest, ZeroLengthRangeProducesNoIterations)
{
    using namespace rllm;
    int count = 0;
    const auto x = static_cast<HeadsIndex>(3);
    for (const auto h : enum_iterator1D<HeadsIndex>(x, x))
        ++count;
    EXPECT_EQ(count, 0);
}

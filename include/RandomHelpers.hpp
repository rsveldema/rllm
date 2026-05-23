#pragma once

#include <cstdlib>
#include <utility>

namespace rllm
{
    /** returns a random value in the range [min, max] */
    inline float get_random_value(float min, float max)
    {
        return min + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (max - min);
    }

    template<typename T>
    T get_random_enum_value(T max_value = T::MAX) {
        static_assert(RAND_MAX >= static_cast<int>(T::MAX), "RAND_MAX must be greater than or equal to the number of enum values");
        return static_cast<T>(rand() % static_cast<int>(max_value));
    }

    template <typename X, typename Y>
    inline std::pair<X, Y> get_random_value_centered_around(X x, Y y, int range = 3)
    {
        int k1 = rand() % (2 * range + 1) - range;
        int k2 = rand() % (2 * range + 1) - range;

        if ((static_cast<int>(x) + k1) < 0)
        {
            k1 = 0;
        }

        if ((static_cast<int>(y) + k2) < 0)
        {
            k2 = 0;
        }

        if ((static_cast<int>(x) + k1) >= static_cast<int>(X::MAX))
        {
            k1 = 0;
        }

        if (static_cast<int>(y) + k2 >= static_cast<int>(Y::MAX))
        {
            k2 = 0;
        }

        return std::make_pair(static_cast<X>(static_cast<int>(x) + k1), static_cast<Y>(static_cast<int>(y) + k2));
    }
} // namespace rllm

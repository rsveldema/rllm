#pragma once

#include <cstdlib>
#include <utility>

namespace rllm
{
    inline size_t random_int(size_t min, size_t max)
    {
        return min + rand() % (max - min + 1);
    }

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

    template<typename T>
    T get_random_enum_value_centered_around(T center, int range) {
        assert(RAND_MAX >= (2 * range + 1)); // sanity check to ensure RAND_MAX is sufficient
        int offset = (rand() % (2 * range + 1)) - range; // random value in [-range, range]
        int raw_value = static_cast<int>(center) + offset;
        int wrapped_value = (raw_value + static_cast<int>(T::MAX)) % static_cast<int>(T::MAX); // wrap around using modulo
        return static_cast<T>(wrapped_value);
    }

} // namespace rllm

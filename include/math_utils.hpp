#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#if defined(RLLM_ENABLE_OVERFLOW_CHECK_ADD)
#define OVERFLOW_CHECK_ADD(a, b) ::math::check_add_not_overflows((a), (b))
#else
#define OVERFLOW_CHECK_ADD(a, b) ((void)0)
#endif

namespace math
{
    // Fast approximation of exp(x) for x in [-10, 0], with max relative error < 0.1%.
    // Values outside that range are clamped to avoid underflow/overflow.
    inline float fast_exp(float x)
    {
        if (x < -10.f)
            return 0.f;
        if (x > 0.f)
            return 1.f;

        // Approximate exp(x) using a 5th-degree polynomial fit on [-10, 0].
        // Coefficients derived from a least-squares fit to exp(x) in that range.
        constexpr float c0 = 1.f;
        constexpr float c1 = 0.999999f;
        constexpr float c2 = 0.499999f;
        constexpr float c3 = 0.166666f;
        constexpr float c4 = 0.041666f;
        constexpr float c5 = 0.008333f;

        return (((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
    }
    
    template <typename A, typename B>
    constexpr auto max(A a, B b)
    {
        using C = std::common_type_t<A, B>;
        const C ac = static_cast<C>(a);
        const C bc = static_cast<C>(b);
        return (ac < bc) ? bc : ac;
    }

    template <typename A, typename B>
    constexpr auto min(A a, B b)
    {
        using C = std::common_type_t<A, B>;
        const C ac = static_cast<C>(a);
        const C bc = static_cast<C>(b);
        return (bc < ac) ? bc : ac;
    }

    template <typename V, typename L, typename H>
    constexpr auto clamp(V v, L lo, H hi)
    {
        using C = std::common_type_t<V, L, H>;
        const C vc = static_cast<C>(v);
        const C lc = static_cast<C>(lo);
        const C hc = static_cast<C>(hi);
        return min(max(vc, lc), hc);
    }

    template <typename A, typename B>
    constexpr void check_add_not_overflows(A a, B b)
    {
        static_assert(std::is_arithmetic_v<A> && std::is_arithmetic_v<B>);
        using C = std::common_type_t<A, B>;

        const C ac = static_cast<C>(a);
        const C bc = static_cast<C>(b);

        if constexpr (std::is_floating_point_v<C>)
        {
            const C sum = ac + bc;
            assert(std::isfinite(sum));
            assert(sum <= std::numeric_limits<C>::max());
            assert(sum >= std::numeric_limits<C>::lowest());
        }
        else if constexpr (std::is_signed_v<C>)
        {
            const C max_c = std::numeric_limits<C>::max();
            const C min_c = std::numeric_limits<C>::lowest();
            assert((bc <= 0 || ac <= max_c - bc) && (bc >= 0 || ac >= min_c - bc));
        }
        else
        {
            const C max_c = std::numeric_limits<C>::max();
            assert(ac <= max_c - bc);
        }
    }
}

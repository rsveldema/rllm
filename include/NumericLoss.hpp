#pragma once

#include <LayerPrimitives.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>

namespace rllm
{
    inline bool parse_numeric_literal(std::string_view literal, long double& value)
    {
        std::string cleaned;
        cleaned.reserve(literal.size());
        for (const char ch : literal)
            if (ch != '_' && ch != '\'')
                cleaned.push_back(ch);

        bool negative = false;
        size_t prefix = 0;
        if (!cleaned.empty() && (cleaned[0] == '+' || cleaned[0] == '-'))
        {
            negative = cleaned[0] == '-';
            prefix = 1;
        }
        if (cleaned.size() > prefix + 2 && cleaned[prefix] == '0' &&
            (cleaned[prefix + 1] == 'b' || cleaned[prefix + 1] == 'B'))
        {
            long double result = 0.0L;
            size_t pos = prefix + 2;
            const size_t digits_begin = pos;
            while (pos < cleaned.size() && (cleaned[pos] == '0' || cleaned[pos] == '1'))
                result = result * 2.0L + static_cast<long double>(cleaned[pos++] - '0');
            if (pos == digits_begin)
                return false;
            value = negative ? -result : result;
            return std::isfinite(value);
        }

        char* end = nullptr;
        value = std::strtold(cleaned.c_str(), &end);
        return end != cleaned.c_str() && std::isfinite(value);
    }

    inline float numeric_distance_cost(std::string_view candidate, std::string_view expected)
    {
        long double candidate_value = 0.0L;
        long double expected_value = 0.0L;
        if (!parse_numeric_literal(candidate, candidate_value) ||
            !parse_numeric_literal(expected, expected_value))
            return 0.0f;

        const auto signed_log = [](long double input) {
            return std::copysign(std::log1p(std::fabs(input)), input);
        };
        const long double difference = std::fabs(
            signed_log(candidate_value) - signed_log(expected_value));
        constexpr long double HUBER_DELTA = 1.0L;
        long double cost = difference <= HUBER_DELTA
            ? 0.5L * difference * difference
            : HUBER_DELTA * (difference - 0.5L * HUBER_DELTA);
        constexpr long double MAX_COST = 8.0L;
        return static_cast<float>(std::min(cost, MAX_COST));
    }
}

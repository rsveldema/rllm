#pragma once

#include <LayerPrimitives.hpp>
#include <cpu/cpu_fixed_matrix.hpp>

#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <memory>

namespace rllm::json_helpers
{
    template <typename Enum>
    constexpr size_t enum_max()
    {
        return static_cast<size_t>(Enum::MAX);
    }

    template <typename Enum, typename T>
    std::unique_ptr<nlohmann::json> serialize_vector(const fixed_size_vector<T, Enum>& v)
    {
        auto out = std::make_unique<nlohmann::json>(nlohmann::json::array());
        for (size_t i = 0; i < enum_max<Enum>(); ++i)
        {
            out->push_back(v[static_cast<Enum>(i)]);
        }
        return out;
    }

    template <typename Enum, typename T>
    void deserialize_vector(const nlohmann::json& j, fixed_size_vector<T, Enum>& v)
    {
        if (!j.is_array() || j.size() != enum_max<Enum>())
        {
            std::fprintf(stderr, "Invalid vector shape in model JSON\n");
            std::abort();
        }

        for (size_t i = 0; i < enum_max<Enum>(); ++i)
        {
            v[static_cast<Enum>(i)] = j.at(i).template get<T>();
        }
    }

    template <typename T, typename X, typename Y>
    std::unique_ptr<nlohmann::json> serialize_matrix(const cpu_fixed_matrix<T, X, Y>& m)
    {
        auto rows = std::make_unique<nlohmann::json>(nlohmann::json::array());

        for (size_t x = 0; x < enum_max<X>(); ++x)
        {
            auto cols = std::make_unique<nlohmann::json>(nlohmann::json::array());
            for (size_t y = 0; y < enum_max<Y>(); ++y)
            {
                cols->push_back(m.get(static_cast<X>(x), static_cast<Y>(y)));
            }
            rows->push_back(std::move(*cols));
        }

        return rows;
    }

    template <typename T, typename X, typename Y>
    void deserialize_matrix(const nlohmann::json& j, cpu_fixed_matrix<T, X, Y>& m)
    {
        if (!j.is_array() || j.size() != enum_max<X>())
        {
            std::fprintf(stderr, "Invalid matrix row count in model JSON\n");
            std::abort();
        }

        for (size_t x = 0; x < enum_max<X>(); ++x)
        {
            const auto& row = j.at(x);
            if (!row.is_array() || row.size() != enum_max<Y>())
            {
                std::fprintf(stderr, "Invalid matrix column count in model JSON\n");
                std::abort();
            }

            for (size_t y = 0; y < enum_max<Y>(); ++y)
            {
                m.set(static_cast<X>(x), static_cast<Y>(y), row.at(y).template get<T>());
            }
        }
    }



} // namespace rllm::json_helpers

namespace rllm::json_helpers
{
} // namespace rllm::json_helpers

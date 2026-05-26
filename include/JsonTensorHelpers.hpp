#pragma once

#include <LayerPrimitives.hpp>

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace rllm::json_helpers
{
    template <typename Enum>
    constexpr size_t enum_max()
    {
        return static_cast<size_t>(Enum::MAX);
    }

    template <typename Enum, typename T>
    nlohmann::json serialize_vector(const fixed_size_vector<T, Enum>& v)
    {
        auto out = nlohmann::json::array();
        for (size_t i = 0; i < enum_max<Enum>(); ++i)
        {
            out.push_back(v[static_cast<Enum>(i)]);
        }
        return out;
    }

    template <typename Enum, typename T>
    void deserialize_vector(const nlohmann::json& j, fixed_size_vector<T, Enum>& v)
    {
        if (!j.is_array() || j.size() != enum_max<Enum>())
        {
            throw std::runtime_error("Invalid vector shape in model JSON");
        }

        for (size_t i = 0; i < enum_max<Enum>(); ++i)
        {
            v[static_cast<Enum>(i)] = j.at(i).template get<T>();
        }
    }

    template <typename T, typename X, typename Y>
    nlohmann::json serialize_matrix(const fixed_size_matrix<T, X, Y>& m)
    {
        auto rows = nlohmann::json::array();

        for (size_t x = 0; x < enum_max<X>(); ++x)
        {
            auto cols = nlohmann::json::array();
            for (size_t y = 0; y < enum_max<Y>(); ++y)
            {
                cols.push_back(m.get(static_cast<X>(x), static_cast<Y>(y)));
            }
            rows.push_back(std::move(cols));
        }

        return rows;
    }

    template <typename T, typename X, typename Y>
    void deserialize_matrix(const nlohmann::json& j, fixed_size_matrix<T, X, Y>& m)
    {
        if (!j.is_array() || j.size() != enum_max<X>())
        {
            throw std::runtime_error("Invalid matrix row count in model JSON");
        }

        for (size_t x = 0; x < enum_max<X>(); ++x)
        {
            const auto& row = j.at(x);
            if (!row.is_array() || row.size() != enum_max<Y>())
            {
                throw std::runtime_error("Invalid matrix column count in model JSON");
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

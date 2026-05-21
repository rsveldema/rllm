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
    nlohmann::json serialize_vector(const template_token_vector<T, Enum>& v)
    {
        auto out = nlohmann::json::array();
        for (size_t i = 0; i < enum_max<Enum>(); ++i)
        {
            out.push_back(v[static_cast<Enum>(i)]);
        }
        return out;
    }

    template <typename Enum, typename T>
    void deserialize_vector(const nlohmann::json& j, template_token_vector<T, Enum>& v)
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
    nlohmann::json serialize_matrix(const template_token_matrix<T, X, Y>& m)
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
    void deserialize_matrix(const nlohmann::json& j, template_token_matrix<T, X, Y>& m)
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



    template <typename X, typename Y>
    nlohmann::json serialize_connection_matrix(
        const template_token_matrix<std::pair<rllm::IntermediateLayerIndex, PositionIndex>, X, Y>& m
    )
    {
        auto rows = nlohmann::json::array();

        for (size_t x = 0; x < rllm::json_helpers::enum_max<X>(); ++x)
        {
            auto cols = nlohmann::json::array();
            for (size_t y = 0; y < rllm::json_helpers::enum_max<Y>(); ++y)
            {
                const auto [target_idx, target_pos] = m.get(static_cast<X>(x), static_cast<Y>(y));
                cols.push_back(nlohmann::json::array(
                    {static_cast<size_t>(target_idx), static_cast<size_t>(target_pos)}
                ));
            }
            rows.push_back(std::move(cols));
        }

        return rows;
    }

    template <typename X, typename Y>
    void deserialize_connection_matrix(
        const nlohmann::json& j,
        template_token_matrix<std::pair<rllm::IntermediateLayerIndex, rllm::PositionIndex>, X, Y>& m
    )
    {
        if (!j.is_array() || j.size() != rllm::json_helpers::enum_max<X>())
        {
            throw std::runtime_error("Invalid connection matrix row count in model JSON");
        }

        for (size_t x = 0; x < rllm::json_helpers::enum_max<X>(); ++x)
        {
            const auto& row = j.at(x);
            if (!row.is_array() || row.size() != rllm::json_helpers::enum_max<Y>())
            {
                throw std::runtime_error("Invalid connection matrix column count in model JSON");
            }

            for (size_t y = 0; y < rllm::json_helpers::enum_max<Y>(); ++y)
            {
                const auto& conn = row.at(y);
                if (!conn.is_array() || conn.size() != 2)
                {
                    throw std::runtime_error("Invalid connection entry in model JSON");
                }

                m.set(
                    static_cast<X>(x),
                    static_cast<Y>(y),
                    std::make_pair(
                        static_cast<rllm::IntermediateLayerIndex>(conn.at(0).template get<size_t>()),
                        static_cast<rllm::PositionIndex>(conn.at(1).template get<size_t>())
                    )
                );
            }
        }
    }

    template <typename X, typename Y>
    nlohmann::json serialize_multi_connection_matrix(
        const template_token_matrix<std::vector<std::pair<rllm::IntermediateLayerIndex, PositionIndex>>, X, Y>& m
    )
    {
        auto rows = nlohmann::json::array();
        for (size_t x = 0; x < rllm::json_helpers::enum_max<X>(); ++x)
        {
            auto cols = nlohmann::json::array();
            for (size_t y = 0; y < rllm::json_helpers::enum_max<Y>(); ++y)
            {
                auto cell = nlohmann::json::array();
                for (const auto& [idx, pos] : m.get(static_cast<X>(x), static_cast<Y>(y)))
                    cell.push_back(nlohmann::json::array({static_cast<size_t>(idx), static_cast<size_t>(pos)}));
                cols.push_back(std::move(cell));
            }
            rows.push_back(std::move(cols));
        }
        return rows;
    }

    template <typename X, typename Y>
    void deserialize_multi_connection_matrix(
        const nlohmann::json& j,
        template_token_matrix<std::vector<std::pair<rllm::IntermediateLayerIndex, rllm::PositionIndex>>, X, Y>& m
    )
    {
        if (!j.is_array() || j.size() != rllm::json_helpers::enum_max<X>())
            throw std::runtime_error("Invalid multi-connection matrix row count in model JSON");

        for (size_t x = 0; x < rllm::json_helpers::enum_max<X>(); ++x)
        {
            const auto& row = j.at(x);
            if (!row.is_array() || row.size() != rllm::json_helpers::enum_max<Y>())
                throw std::runtime_error("Invalid multi-connection matrix column count in model JSON");

            for (size_t y = 0; y < rllm::json_helpers::enum_max<Y>(); ++y)
            {
                const auto& cell = row.at(y);
                if (!cell.is_array())
                    throw std::runtime_error("Invalid multi-connection cell in model JSON");

                std::vector<std::pair<rllm::IntermediateLayerIndex, rllm::PositionIndex>> conns;
                conns.reserve(cell.size());
                for (const auto& conn : cell)
                {
                    if (!conn.is_array() || conn.size() != 2)
                        throw std::runtime_error("Invalid connection entry in model JSON");
                    conns.emplace_back(
                        static_cast<rllm::IntermediateLayerIndex>(conn.at(0).template get<size_t>()),
                        static_cast<rllm::PositionIndex>(conn.at(1).template get<size_t>())
                    );
                }
                m.set(static_cast<X>(x), static_cast<Y>(y), std::move(conns));
            }
        }
    }

} // namespace rllm::json_helpers

namespace rllm::json_helpers
{
    template <typename Enum>
    nlohmann::json serialize_multi_connection_vector(
        const template_token_vector<std::vector<rllm::IntermediateLayerIndex>, Enum>& v
    )
    {
        auto out = nlohmann::json::array();
        for (size_t x = 0; x < enum_max<Enum>(); ++x)
        {
            auto cell = nlohmann::json::array();
            for (const auto& idx : v[static_cast<Enum>(x)])
                cell.push_back(static_cast<size_t>(idx));
            out.push_back(std::move(cell));
        }
        return out;
    }

    template <typename Enum>
    void deserialize_multi_connection_vector(
        const nlohmann::json& j,
        template_token_vector<std::vector<rllm::IntermediateLayerIndex>, Enum>& v
    )
    {
        if (!j.is_array() || j.size() != enum_max<Enum>())
            throw std::runtime_error("Invalid multi-connection vector size in model JSON");
        for (size_t x = 0; x < enum_max<Enum>(); ++x)
        {
            const auto& cell = j.at(x);
            if (!cell.is_array())
                throw std::runtime_error("Invalid multi-connection cell in model JSON");
            std::vector<rllm::IntermediateLayerIndex> conns;
            conns.reserve(cell.size());
            for (const auto& entry : cell)
                conns.push_back(static_cast<rllm::IntermediateLayerIndex>(entry.template get<size_t>()));
            v[static_cast<Enum>(x)] = std::move(conns);
        }
    }

} // namespace rllm::json_helpers

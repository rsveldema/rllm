#include <RLLM.hpp>

#include <fstream>
#include <nlohmann/json.hpp>
#include <print>
#include <stdexcept>

namespace
{

    template <typename Enum>
    constexpr size_t enum_max()
    {
        return static_cast<size_t>(Enum::MAX);
    }

    template <typename Enum, typename T>
    nlohmann::json serialize_vector(const rllm::template_token_vector<T, Enum>& v)
    {
        auto out = nlohmann::json::array();
        for (size_t i = 0; i < enum_max<Enum>(); ++i)
        {
            out.push_back(v[static_cast<Enum>(i)]);
        }
        return out;
    }

    template <typename Enum, typename T>
    void deserialize_vector(const nlohmann::json& j, rllm::template_token_vector<T, Enum>& v)
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
    nlohmann::json serialize_matrix(const rllm::template_token_matrix<T, X, Y>& m)
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
    void deserialize_matrix(const nlohmann::json& j, rllm::template_token_matrix<T, X, Y>& m)
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
        const rllm::template_token_matrix<std::pair<rllm::IntermediateLayerIndex, rllm::PositionIndex>, X, Y>& m
    )
    {
        auto rows = nlohmann::json::array();

        for (size_t x = 0; x < enum_max<X>(); ++x)
        {
            auto cols = nlohmann::json::array();
            for (size_t y = 0; y < enum_max<Y>(); ++y)
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
        rllm::template_token_matrix<std::pair<rllm::IntermediateLayerIndex, rllm::PositionIndex>, X, Y>& m
    )
    {
        if (!j.is_array() || j.size() != enum_max<X>())
        {
            throw std::runtime_error("Invalid connection matrix row count in model JSON");
        }

        for (size_t x = 0; x < enum_max<X>(); ++x)
        {
            const auto& row = j.at(x);
            if (!row.is_array() || row.size() != enum_max<Y>())
            {
                throw std::runtime_error("Invalid connection matrix column count in model JSON");
            }

            for (size_t y = 0; y < enum_max<Y>(); ++y)
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

} // namespace

namespace rllm
{

    void InputLayer::load(const nlohmann::json& j)
    {
        deserialize_matrix(j.at("trigger_values"), m_trigger_values);
        deserialize_matrix(j.at("weights"), m_weights);
        deserialize_connection_matrix(j.at("connections"), m_connections);
        m_inputs.fill(0.0f);
    }

    nlohmann::json InputLayer::save() const
    {
        return {
            {"trigger_values", serialize_matrix(m_trigger_values)},
            {"weights", serialize_matrix(m_weights)},
            {"connections", serialize_connection_matrix(m_connections)}
        };
    }

    void IntermediateLayer::load(const nlohmann::json& j)
    {
        deserialize_matrix(j.at("trigger_values"), m_trigger_values);
        deserialize_matrix(j.at("weights"), m_weights);
        deserialize_connection_matrix(j.at("connections"), m_connections);
        m_inputs.fill(0.0f);
    }

    nlohmann::json IntermediateLayer::save() const
    {
        return {
            {"trigger_values", serialize_matrix(m_trigger_values)},
            {"weights", serialize_matrix(m_weights)},
            {"connections", serialize_connection_matrix(m_connections)}
        };
    }

    void OutputLayer::load(const nlohmann::json& j)
    {
        deserialize_vector(j.at("trigger_values"), m_trigger_values);
        deserialize_vector(j.at("weights"), m_weights);
        m_inputs.fill(0.0f);
    }

    nlohmann::json OutputLayer::save() const
    {
        return {
            {"trigger_values", serialize_vector(m_trigger_values)},
            {"weights", serialize_vector(m_weights)}
        };
    }

    void NeuralNetwork::load(const std::string& filename)
    {
        try
        {
            std::ifstream file{filename};
            if (!file)
                return;

            const auto j = nlohmann::json::parse(file);

            const auto version = j.value("version", 0);
            if (version != 1)
            {
                throw std::runtime_error("Unsupported model version");
            }

            m_input_layer.load(j.at("input_layer"));

            const auto& layers_j = j.at("intermediate_layers");
            if (!layers_j.is_array())
            {
                throw std::runtime_error("intermediate_layers must be an array");
            }

            if (layers_j.size() != m_intermediate_layers.size())
            {
                throw std::runtime_error("Loaded model layer count does not match network configuration");
            }

            for (size_t i = 0; i < layers_j.size(); ++i)
            {
                auto& layer = m_intermediate_layers[i];
                layer.load(layers_j.at(i));
            }

            m_output_layer.load(j.at("output_layer"));
        }
        catch (const std::exception& e)
        {
            std::println("Failed to load model '{}': {}", filename, e.what());
        }
    }

    void NeuralNetwork::save(const std::string& filename) const
    {
        nlohmann::json j;
        j["version"] = 1;

        j["input_layer"] = m_input_layer.save();

        auto layers = nlohmann::json::array();
        for (const auto& layer : m_intermediate_layers)
        {
            layers.push_back(layer.save());
        }
        j["intermediate_layers"] = std::move(layers);

        j["output_layer"] = m_output_layer.save();

        std::ofstream file{filename};
        file << j.dump(2) << '\n';
    }

    void Corpus::save_token_map(const std::string& filename) const
    {
        nlohmann::json j;
        for (const auto& [token, id] : m_token_to_id)
        {
            j[token] = id;
        }
        std::ofstream file{filename};
        file << j.dump(2) << '\n';
    }


} // namespace rllm

#include <RLLM.hpp>
#include <JsonTensorHelpers.hpp>

#include <fstream>
#include <nlohmann/json.hpp>
#include <print>
#include <stdexcept>


namespace rllm
{

    void InputLayer::load(const nlohmann::json& j)
    {
        json_helpers::deserialize_matrix(j.at("trigger_values"), m_trigger_values);
        json_helpers::deserialize_matrix(j.at("weights"), m_weights);
        json_helpers::deserialize_connection_matrix(j.at("connections"), m_connections);
        m_inputs.fill(0.0f);
    }

    nlohmann::json InputLayer::save() const
    {
        return {
            {"trigger_values", json_helpers::serialize_matrix(m_trigger_values)},
            {"weights", json_helpers::serialize_matrix(m_weights)},
            {"connections", json_helpers::serialize_connection_matrix(m_connections)}
        };
    }

    void OutputLayer::load(const nlohmann::json& j)
    {
        json_helpers::deserialize_vector(j.at("trigger_values"), m_trigger_values);
        m_inputs.fill(0.0f);
    }

    nlohmann::json OutputLayer::save() const
    {
        return {
            {"trigger_values", json_helpers::serialize_vector(m_trigger_values)}
        };
    }

    void IntermediateLayer::load(const nlohmann::json& j)
    {
        json_helpers::deserialize_matrix(j.at("trigger_values"), m_trigger_values);
        json_helpers::deserialize_matrix(j.at("weights"), m_weights);
        json_helpers::deserialize_connection_matrix(j.at("connections"), m_connections);
        m_inputs.fill(0.0f);
    }

    nlohmann::json IntermediateLayer::save() const
    {
        return {
            {"trigger_values", json_helpers::serialize_matrix(m_trigger_values)},
            {"weights", json_helpers::serialize_matrix(m_weights)},
            {"connections", json_helpers::serialize_connection_matrix(m_connections)}
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
} // namespace rllm

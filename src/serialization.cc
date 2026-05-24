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
        if (!j.contains("embeddings")) return; // backwards compat: skip if absent

        const auto& emb_j = j.at("embeddings");
        for (size_t t = 0; t < static_cast<size_t>(TokenID::MAX); ++t)
        {
            const auto& row = emb_j.at(t);
            for (size_t d = 0; d < static_cast<size_t>(EmbeddingDimension::MAX); ++d)
                m_embeddings[static_cast<TokenID>(t)][static_cast<EmbeddingDimension>(d)] = row.at(d).template get<float>();
        }
    }

    nlohmann::json InputLayer::save() const
    {
        auto emb_j = nlohmann::json::array();
        for (size_t t = 0; t < static_cast<size_t>(TokenID::MAX); ++t)
        {
            auto row = nlohmann::json::array();
            for (size_t d = 0; d < static_cast<size_t>(EmbeddingDimension::MAX); ++d)
                row.push_back(m_embeddings[static_cast<TokenID>(t)][static_cast<EmbeddingDimension>(d)]);
            emb_j.push_back(std::move(row));
        }
        return {{"embeddings", std::move(emb_j)}};
    }

    void OutputLayer::load(const nlohmann::json& )
    {
        m_inputs.fill(0.0f);
    }

    nlohmann::json OutputLayer::save() const
    {
        return {
        };
    }

    void IntermediateLayer::load(const nlohmann::json& j)
    {
        m_inputs.fill(0.0f);

        const auto& conns_j = j.at("connections");
        if (!conns_j.is_array() || conns_j.size() != static_cast<size_t>(IntermediateLayerIndex::MAX))
            throw std::runtime_error("connections array has wrong size");

        for (size_t i = 0; i < static_cast<size_t>(IntermediateLayerIndex::MAX); ++i)
        {
            const auto idx = static_cast<IntermediateLayerIndex>(i);
            auto& vec = m_connections[idx];
            vec.clear();
            for (const auto& c : conns_j.at(i))
            {
                vec.push_back({
                    .target_neuron = static_cast<IntermediateLayerIndex>(c.at("target").template get<size_t>()),
                    .weight        = c.at("weight").template get<float>()
                });
            }
        }

        // attention gate weights (optional for backwards compatibility)
        if (j.contains("attn_weights"))
        {
            const auto& aw_j = j.at("attn_weights");
            for (size_t i = 0; i < static_cast<size_t>(IntermediateLayerIndex::MAX); ++i)
                m_attn_weights[static_cast<IntermediateLayerIndex>(i)] = aw_j.at(i).template get<float>();
        }
        else
        {
            // initialise to 1 so existing checkpoints behave as before (gate = sigmoid(x))
            m_attn_weights.fill(1.0f);
        }
        m_attn_vel.fill(0.0f);
    }

    nlohmann::json IntermediateLayer::save() const
    {
        auto conns_j = nlohmann::json::array();
        for (size_t i = 0; i < static_cast<size_t>(IntermediateLayerIndex::MAX); ++i)
        {
            const auto idx = static_cast<IntermediateLayerIndex>(i);
            auto neuron_conns = nlohmann::json::array();
            for (const auto ci : enum_iterator<NeuronConnectionIndex>(m_connections[idx].size()))
            {
                const auto& c = m_connections[idx][ci];
                neuron_conns.push_back({
                    {"target", static_cast<size_t>(c.target_neuron)},
                    {"weight", c.weight}
                });
            }
            conns_j.push_back(std::move(neuron_conns));
        }
        return {{
            "connections", std::move(conns_j)},
            {"attn_weights", [this]{
                auto aw = nlohmann::json::array();
                for (size_t i = 0; i < static_cast<size_t>(IntermediateLayerIndex::MAX); ++i)
                    aw.push_back(m_attn_weights[static_cast<IntermediateLayerIndex>(i)]);
                return aw;
            }()}
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

            m_intermediate_layers.clear();
            m_intermediate_layers.reserve(layers_j.size());
            for (const auto& layer_j : layers_j)
            {
                m_intermediate_layers.emplace_back(m_corpus);
                m_intermediate_layers.back().load(layer_j);
            }

            m_output_layer.load(j.at("output_layer"));
        }
        catch (const std::exception& e)
        {
            std::println("Failed to load model '{}': {}", filename, e.what());
            std::abort();
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

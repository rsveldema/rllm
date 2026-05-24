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

    void OutputLayer::load(const nlohmann::json& j)
    {
        m_inputs.fill(0.0f);
        if (j.contains("W_lm_head"))
        {
            const auto& w_j = j.at("W_lm_head");
            W_lm_head.resize(w_j.size());
            for (size_t i = 0; i < W_lm_head.size(); ++i)
                W_lm_head[i] = w_j.at(i).template get<float>();
        }
        V_lm_head.assign(W_lm_head.size(), 0.f);
    }

    nlohmann::json OutputLayer::save() const
    {
        auto w_j = nlohmann::json::array();
        w_j.get_ref<nlohmann::json::array_t&>().reserve(W_lm_head.size());
        for (float v : W_lm_head) w_j.push_back(v);
        return {{"W_lm_head", std::move(w_j)}};
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

            if (j.contains("transformer_blocks"))
            {
                const auto& blocks_j = j.at("transformer_blocks");
                m_transformer_blocks.clear();
                m_transformer_blocks.reserve(blocks_j.size());
                for (const auto& b : blocks_j)
                {
                    m_transformer_blocks.emplace_back();
                    m_transformer_blocks.back().load(b);
                }
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

        auto blocks = nlohmann::json::array();
        for (const auto& block : m_transformer_blocks)
            blocks.push_back(block.save());
        j["transformer_blocks"] = std::move(blocks);

        j["output_layer"] = m_output_layer.save();

        std::ofstream file{filename};
        file << j.dump(2) << '\n';
    }
} // namespace rllm

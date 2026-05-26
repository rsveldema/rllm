#include <RLLM.hpp>
#include <JsonTensorHelpers.hpp>

#include <fstream>
#include <format>
#include <nlohmann/json.hpp>
#include <print>
#include <stdexcept>
#include <string>


namespace rllm
{
    namespace
    {
        // Stable checksum of tokenizer contents to detect model/tokenizer mismatch.
        uint64_t tokenizer_signature()
        {
            uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis
            for (const auto& [id, info] : tokenizer_map)
            {
                const auto id_val = static_cast<uint64_t>(static_cast<int32_t>(id));
                h ^= id_val;
                h *= 1099511628211ULL;

                for (const char* p = info.str; *p != '\0'; ++p)
                {
                    h ^= static_cast<uint64_t>(static_cast<unsigned char>(*p));
                    h *= 1099511628211ULL;
                }

                h ^= static_cast<uint64_t>(info.end_of_word ? 1 : 0);
                h *= 1099511628211ULL;
            }
            return h;
        }
    } // namespace


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
        m_inputs.fill(RLMM_ZERO);
        if (j.contains("W_lm_head"))
        {
            const auto& w_j = j.at("W_lm_head");
            size_t i = 0;
            for (const auto t : enum_iterator<TokenID>())
                for (const auto d : enum_iterator<EmbeddingDimension>())
                    W_lm_head[t, d] = w_j.at(i++).template get<float>();
        }
        V_lm_head.fill(RLMM_ZERO);
    }

    nlohmann::json OutputLayer::save() const
    {
        auto w_j = nlohmann::json::array();
        w_j.get_ref<nlohmann::json::array_t&>().reserve(W_lm_head.ROWS * W_lm_head.COLS);
        for (const auto t : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                w_j.push_back(W_lm_head[t, d]);
        return {{"W_lm_head", std::move(w_j)}};
    }


    bool NeuralNetwork::load(const std::string& filename)
    {
        try
        {
            std::ifstream file{filename};
            if (!file) {
                return false;
            }

            const auto j = nlohmann::json::parse(file);

            const auto version = j.value("version", 0);
            if (version != 1)
            {
                throw std::runtime_error("Unsupported model version");
            }

            const auto expected_vocab_size = static_cast<size_t>(TokenID::MAX);
            if (j.contains("tokenizer_vocab_size"))
            {
                const auto model_vocab_size = j.at("tokenizer_vocab_size").template get<size_t>();
                if (model_vocab_size != expected_vocab_size)
                {
                    throw std::runtime_error(
                        std::format(
                            "Tokenizer vocab size mismatch (model={}, runtime={}). "
                            "Rebuild/regenerate tokenizer map and retrain or load a compatible model.",
                            model_vocab_size,
                            expected_vocab_size
                        )
                    );
                }
            }

            if (j.contains("tokenizer_signature"))
            {
                const auto model_sig = j.at("tokenizer_signature").template get<uint64_t>();
                const auto runtime_sig = tokenizer_signature();
                if (model_sig != runtime_sig)
                {
                    throw std::runtime_error(
                        "Tokenizer signature mismatch between model and runtime tokenizer map. "
                        "Rebuild/regenerate tokenizer map and retrain or load a compatible model."
                    );
                }
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
        return true;
    }

    void NeuralNetwork::save(const std::string& filename) const
    {
        nlohmann::json j;
        j["version"] = 1;
        j["tokenizer_vocab_size"] = static_cast<size_t>(TokenID::MAX);
        j["tokenizer_signature"] = tokenizer_signature();

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

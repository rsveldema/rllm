#include <JsonTensorHelpers.hpp>
#include <Trainer.hpp>
#include <safetensors.hh>

#include <chrono>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <print>
#include <string>
#include <string_view>


namespace rllm
{
    namespace
    {
        bool is_safetensors_model_filename(std::string_view filename)
        {
            const auto extension_pos = filename.rfind('.');
            if (extension_pos == std::string_view::npos)
                return false;

            const auto extension = filename.substr(extension_pos);
            return extension == ".st" || extension == ".safetensors";
        }

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
        if (!j.contains("embeddings"))
            return;

        json_helpers::deserialize_matrix(j.at("embeddings"), m_embeddings_cpu);
        m_embeddings.copy_from_cpu(m_embeddings_cpu);
    }

    nlohmann::json InputLayer::save() const
    {
        return {{"embeddings", *json_helpers::serialize_matrix(m_embeddings_cpu)}};
    }

    void OutputLayer::load(const nlohmann::json& j)
    {
        m_inputs.zero();
        if (j.contains("W_lm_head"))
        {
            const auto& w_j = j.at("W_lm_head");
            cpu_fixed_matrix<float16, TokenID, EmbeddingDimension> cpu_tmp;
            size_t i = 0;
            for (const auto t : enum_iterator1D<TokenID>())
                for (const auto d : enum_iterator1D<EmbeddingDimension>())
                    cpu_tmp.set(t, d, static_cast<float16>(w_j.at(i++).template get<float>()));
            W_lm_head.copy_from_cpu(cpu_tmp);
        }
        V_lm_head.zero();
    }

    nlohmann::json OutputLayer::save() const
    {
        cpu_fixed_matrix<float16, TokenID, EmbeddingDimension> cpu_tmp;
        W_lm_head.copy_to_cpu(cpu_tmp);
        auto w_j = nlohmann::json::array();
        w_j.get_ref<nlohmann::json::array_t&>().reserve(W_lm_head.ROWS * W_lm_head.COLS);
        for (const auto t : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                w_j.push_back(static_cast<float>(cpu_tmp.get(t, d)));
        return {{"W_lm_head", std::move(w_j)}};
    }


    bool NeuralNetwork::load(const std::string& filename)
    {
        if (is_safetensors_model_filename(filename))
            return load_from_safetensors(filename);

        if (filename.empty() || filename.size() < 5 || filename.compare(filename.size() - 5, 5, ".json") != 0)
        {
            std::println("Unknown model format: '{}'", filename);
            return false;
        }

        std::ifstream file{filename};
        if (!file)
            return false;

        const auto j = nlohmann::json::parse(file);
        const auto version = j.value("version", 0);
        if (version != 1)
            std::println("Unsupported model version in {}", filename);
        std::abort();

        const auto expected_vocab_size = static_cast<size_t>(TokenID::MAX);
        if (j.contains("tokenizer_vocab_size"))
        {
            const auto model_vocab_size = j.at("tokenizer_vocab_size").template get<size_t>();
            if (model_vocab_size != expected_vocab_size)
                std::println("Tokenizer vocab size mismatch in {}: model={}, runtime={}", filename, model_vocab_size, expected_vocab_size);
            std::abort();
        }

        if (j.contains("tokenizer_signature"))
        {
            const auto model_sig = j.at("tokenizer_signature").template get<uint64_t>();
            const auto runtime_sig = tokenizer_signature();
            if (model_sig != runtime_sig)
                std::println("Tokenizer signature mismatch in {}: rebuilding/retraining required", filename);
            std::abort();
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

        if (j.contains("output_layers"))
        {
            const auto& output_layers_j = j.at("output_layers");
            const size_t n = std::min(output_layers_j.size(), static_cast<size_t>(MultiTokenPredictionIndex::MAX));
            for (size_t i = 0; i < n; ++i)
                m_output_layers[static_cast<MultiTokenPredictionIndex>(i)].load(output_layers_j.at(i));
        }
        return true;
    }

    void NeuralNetwork::save(const std::string& filename) const
    {
        if (is_safetensors_model_filename(filename))
        {
            save_to_safetensors(filename);
            return;
        }

        if (filename.empty() || filename.size() < 5 || filename.compare(filename.size() - 5, 5, ".json") != 0)
        {
            std::println("Unknown model format: '{}'", filename);
            return;
        }

        const auto save_start = std::chrono::steady_clock::now();

        nlohmann::json j;
        j["version"] = 1;
        j["tokenizer_vocab_size"] = static_cast<size_t>(TokenID::MAX);
        j["tokenizer_signature"] = tokenizer_signature();
        j["input_layer"] = m_input_layer.save();

        auto blocks = nlohmann::json::array();
        for (const auto& block : m_transformer_blocks)
            blocks.push_back(*block.save());
        j["transformer_blocks"] = std::move(blocks);

        auto output_layers = nlohmann::json::array();
        for (const auto idx : enum_iterator1D<MultiTokenPredictionIndex>())
            output_layers.push_back(m_output_layers[idx].save());
        j["output_layers"] = std::move(output_layers);

        std::ofstream file{filename};
        file << j.dump(2) << '\n';

        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - save_start).count();
        std::println("Saved model '{}' in {:.3f} seconds", filename, elapsed);
    }

    void NeuralNetwork::checkpoint() const
    {
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        save(std::format("models/checkpoint-{}.st", now_ms));
    }


    // ============================================================================
    // Safetensors helpers
    // ============================================================================

    namespace
    {
        safetensors::safetensors_t make_safetensors_metadata()
        {
            safetensors::safetensors_t st;
            st.metadata.insert("version", "2");
            st.metadata.insert("creator", "rllm");
            return st;
        }

        template <typename T, typename X, typename Y>
        void push_matrix(safetensors::safetensors_t& st, const std::string& key, const fixed_size_matrix<T, X, Y>& m, std::vector<uint8_t>& storage)
        {
            // D2H: download GPU matrix to cpu intermediate first
            cpu_fixed_matrix<T, X, Y> cpu_tmp;
            m.copy_to_cpu(cpu_tmp);

            safetensors::tensor_t tensor;
            tensor.dtype = safetensors::dtype::kFLOAT32;
            size_t rows = static_cast<size_t>(X::MAX);
            size_t cols = static_cast<size_t>(Y::MAX);
            tensor.shape.resize(2);
            tensor.shape[0] = rows;
            tensor.shape[1] = cols;

            size_t n = rows * cols;
            size_t data_size = n * sizeof(float);
            size_t dst_offset = storage.size();

            std::vector<float> flat(n);
            for (size_t x = 0; x < rows; ++x)
                for (size_t y = 0; y < cols; ++y)
                    flat[x * cols + y] = static_cast<float>(cpu_tmp.get(static_cast<X>(x), static_cast<Y>(y)));

            storage.resize(storage.size() + data_size);
            std::copy(flat.begin(), flat.end(), reinterpret_cast<float*>(storage.data() + dst_offset));

            tensor.data_offsets[0] = dst_offset;
            tensor.data_offsets[1] = dst_offset + data_size;
            st.tensors.insert(key, tensor);
        }

        template <typename T, typename X, typename Y>
        void pull_matrix(const std::string& key, const safetensors::safetensors_t& st, fixed_size_matrix<T, X, Y>& m)
        {
            safetensors::tensor_t tensor;
            if (!st.tensors.at(key, &tensor))
            {
                std::println("Missing tensor {} in safetensors file", key);
                std::abort();
            }
            if (tensor.dtype != safetensors::dtype::kFLOAT32)
            {
                std::println("Expected kFLOAT32 for: {}", key);
                std::abort();
            }

            size_t rows = static_cast<size_t>(X::MAX);
            size_t cols = static_cast<size_t>(Y::MAX);
            if (tensor.shape.size() != 2 || tensor.shape[0] != rows || tensor.shape[1] != cols)
            {
                std::println("Shape mismatch for: {} (expected {{{}, {}}})", key, rows, cols);
                std::abort();
            }

            const uint8_t* base = st.storage.empty() ? st.databuffer_addr : reinterpret_cast<const uint8_t*>(st.storage.data());
            assert(base != nullptr);
            auto* data = reinterpret_cast<float const*>(base + tensor.data_offsets[0]);
            assert(data != nullptr);

            // Fill cpu intermediate then H2D upload
            cpu_fixed_matrix<T, X, Y> cpu_tmp;
            for (size_t x = 0; x < rows; ++x)
                for (size_t y = 0; y < cols; ++y)
                    cpu_tmp.set(static_cast<X>(x), static_cast<Y>(y), static_cast<T>(data[x * cols + y]));
            m.copy_from_cpu(cpu_tmp);
        }

    } // anonymous namespace


    // ---- InputLayer -----------------------------------------------------------

    void InputLayer::load_from_safetensors(const std::string& filename, std::string* err)
    {
        safetensors::safetensors_t st;
        if (!safetensors::load_from_file(filename, &st, nullptr, err))
        {
            if (err)
                *err = "Failed to read safetensors: " + filename;
            return;
        }

        pull_matrix("input_layer.embeddings", st, m_embeddings);
        m_embeddings.copy_to_cpu(m_embeddings_cpu);
    }

    void InputLayer::save_to_safetensors(const std::string& filename, std::string* warn, std::string* err) const
    {
        (void) warn;
        safetensors::safetensors_t st = make_safetensors_metadata();
        std::vector<uint8_t> storage;

        push_matrix(st, "input_layer.embeddings", m_embeddings, storage);
        st.storage = std::move(storage);

        if (!safetensors::save_to_file(st, filename, warn, err))
            if (err)
                *err = "Failed to write safetensors: " + filename;
    }


    // ---- TransformerBlock -----------------------------------------------------

    void TransformerBlock::load_from_safetensors(const std::string& filename, std::string* err)
    {
        safetensors::safetensors_t st;
        if (!safetensors::load_from_file(filename, &st, nullptr, err))
        {
            if (err)
                *err = "Failed to read safetensors: " + filename;
            return;
        }

        // filename contains the prefix "transformer_blocks.<i>." passed by caller.
        const std::string pfx = filename;
        pull_matrix(pfx + "W_q", st, W_q);
        pull_matrix(pfx + "W_k", st, W_k);
        pull_matrix(pfx + "W_v", st, W_v);
        pull_matrix(pfx + "W_o", st, W_o);
        pull_matrix(pfx + "W_gate", st, W_gate);
        pull_matrix(pfx + "W_up", st, W_up);
        pull_matrix(pfx + "W_down", st, W_down);

        copy_weights_to_offload_buffer();
        V_q.zero();
        V_k.zero();
        V_v.zero();
        V_o.zero();
        V_gate.zero();
        V_up.zero();
        V_down.zero();
    }

    void TransformerBlock::save_to_safetensors(const std::string& filename, std::string* warn, std::string* err) const
    {
        (void) warn;
        safetensors::safetensors_t st = make_safetensors_metadata();
        std::vector<uint8_t> storage;

        // filename parameter contains the caller-provided prefix (e.g., "transformer_blocks.0.").
        const std::string pfx = filename;
        push_matrix(st, pfx + "W_q", W_q, storage);
        push_matrix(st, pfx + "W_k", W_k, storage);
        push_matrix(st, pfx + "W_v", W_v, storage);
        push_matrix(st, pfx + "W_o", W_o, storage);
        push_matrix(st, pfx + "W_gate", W_gate, storage);
        push_matrix(st, pfx + "W_up", W_up, storage);
        push_matrix(st, pfx + "W_down", W_down, storage);

        st.storage = std::move(storage);

        if (!safetensors::save_to_file(st, filename, warn, err))
            if (err)
                *err = "Failed to write safetensors: " + filename;
    }


    // ---- OutputLayer ------------------------------------------------------------

    void OutputLayer::load_from_safetensors(const std::string& filename, std::string* err)
    {
        safetensors::safetensors_t st;
        if (!safetensors::load_from_file(filename, &st, nullptr, err))
        {
            if (err)
                *err = "Failed to read safetensors: " + filename;
            return;
        }

        pull_matrix("output_layers.W_lm_head", st, W_lm_head);
        m_inputs.zero();
        V_lm_head.zero();
    }

    void OutputLayer::save_to_safetensors(const std::string& filename, std::string* warn, std::string* err) const
    {
        (void) warn;
        safetensors::safetensors_t st = make_safetensors_metadata();
        std::vector<uint8_t> storage;

        push_matrix(st, "output_layers.W_lm_head", W_lm_head, storage);
        st.storage = std::move(storage);

        if (!safetensors::save_to_file(st, filename, warn, err))
            if (err)
                *err = "Failed to write safetensors: " + filename;
    }


    // ---- NeuralNetwork-level save/load for checkpointing ----------------------

    void NeuralNetwork::save_to_safetensors(const std::string& filename) const
    {
        safetensors::safetensors_t st = make_safetensors_metadata();
        std::vector<uint8_t> storage;

        // Input layer (friend grants access to m_embeddings)
        push_matrix(st, "input_layer.embeddings", m_input_layer.m_embeddings, storage);

        // Transformer blocks (friend grants access to W_q etc.)
        for (size_t bi = 0; bi < m_transformer_blocks.size(); ++bi)
        {
            const auto& block = m_transformer_blocks[bi];
            std::string pfx = "transformer_blocks." + std::to_string(bi) + ".";
            push_matrix(st, pfx + "W_q", block.W_q, storage);
            push_matrix(st, pfx + "W_k", block.W_k, storage);
            push_matrix(st, pfx + "W_v", block.W_v, storage);
            push_matrix(st, pfx + "W_o", block.W_o, storage);
            push_matrix(st, pfx + "W_gate", block.W_gate, storage);
            push_matrix(st, pfx + "W_up", block.W_up, storage);
            push_matrix(st, pfx + "W_down", block.W_down, storage);
        }

        // Output layer (friend grants access to m_output_layers internals)
        std::string out_key = "output_layers.W_lm_head";
        constexpr auto last_output_index = static_cast<MultiTokenPredictionIndex>(static_cast<size_t>(MultiTokenPredictionIndex::MAX) - 1);
        auto& last_out = m_output_layers[last_output_index];
        push_matrix(st, out_key, last_out.W_lm_head, storage);

        // Tokenizer metadata for validation on load.
        st.metadata.insert("tokenizer_vocab_size", std::to_string(static_cast<size_t>(TokenID::MAX)));
        st.metadata.insert("tokenizer_signature", std::to_string(tokenizer_signature()));

        st.storage = std::move(storage);
        std::string warn;
        if (!safetensors::save_to_file(st, filename, &warn, nullptr))
        {
            std::println("Failed to write safetensors file: {}", filename);
            std::abort();
        }
    }

    bool NeuralNetwork::load_from_safetensors(const std::string& filename)
    {
        safetensors::safetensors_t st;
        std::string warn, ere;
        if (!safetensors::load_from_file(filename, &st, &warn, &ere))
        {
            std::println("Failed to read safetensors file: {}", filename);
            std::abort();
        }

        const auto& header = st.metadata;
        if (!header.count("version"))
        {
            std::println("Missing version in safetensors metadata for: {}", filename);
            std::abort();
        }

        // Validate tokenizer if present (matches JSON load behavior).
        std::string sig_str;
        if (header.count("tokenizer_signature") && header.at("tokenizer_signature", &sig_str))
        {
            const auto model_sig = static_cast<uint64_t>(std::stoull(sig_str));
            const auto runtime_sig = tokenizer_signature();
            if (model_sig != runtime_sig)
            {
                std::println("Tokenizer signature mismatch between safetensors model and runtime tokenizer for: {}", filename);
                std::abort();
            }
        }

        // Count transformer blocks by counting W_q tensors.
        size_t num_blocks = 0;
        while (st.tensors.count("transformer_blocks." + std::to_string(num_blocks) + ".W_q"))
            ++num_blocks;

        m_input_layer.load_from_safetensors(filename);

        // Clear existing blocks and create new ones.
        m_transformer_blocks.clear();
        m_transformer_blocks.reserve(num_blocks);

        for (size_t bi = 0; bi < num_blocks; ++bi)
        {
            std::string pfx = "transformer_blocks." + std::to_string(bi) + ".";

            auto& block = m_transformer_blocks.emplace_back();

            // Directly pull each weight matrix into the block's members.
            pull_matrix(pfx + "W_q", st, block.W_q);
            pull_matrix(pfx + "W_k", st, block.W_k);
            pull_matrix(pfx + "W_v", st, block.W_v);
            pull_matrix(pfx + "W_o", st, block.W_o);
            pull_matrix(pfx + "W_gate", st, block.W_gate);
            pull_matrix(pfx + "W_up", st, block.W_up);
            pull_matrix(pfx + "W_down", st, block.W_down);

            block.copy_weights_to_offload_buffer();
        }

        if (st.tensors.count("output_layers.W_lm_head"))
            for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
                m_output_layers[oi].load_from_safetensors(filename);

        return true;
    }

} // namespace rllm

#include <JsonTensorHelpers.hpp>
#include <Trainer.hpp>
#include <safetensors.hh>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
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

        bool parse_metadata_u64(const std::string& text, uint64_t& value)
        {
            const auto* begin = text.data();
            const auto* end = begin + text.size();
            const auto result = std::from_chars(begin, end, value);
            return result.ec == std::errc{} && result.ptr == end;
        }

        template <typename T>
        bool parse_metadata_number(const std::string& text, T& value)
        {
            const auto* begin = text.data();
            const auto* end = begin + text.size();
            const auto result = std::from_chars(begin, end, value);
            return result.ec == std::errc{} && result.ptr == end;
        }
    } // namespace


    void InputLayer::load(const nlohmann::json& j)
    {
        m_adam_first.zero();
        m_adam_second.zero();
        if (!j.contains("embeddings"))
            return;

        json_helpers::deserialize_matrix(j.at("embeddings"), m_embeddings_cpu);
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_embeddings.copy_from_cpu(queue, m_embeddings_cpu);
    }

    nlohmann::json InputLayer::save() const
    {
        return {{"embeddings", *json_helpers::serialize_matrix(m_embeddings_cpu)}};
    }

    void OutputLayer::load(const nlohmann::json& j)
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_inputs.zero(queue);
        if (j.contains("W_lm_head"))
        {
            cpu_fixed_matrix<float16, TokenID, EmbeddingDimension> w_lm_head_cpu;
            const auto& w_j = j.at("W_lm_head");
            size_t i = 0;
            for (const auto t : enum_iterator1D<TokenID>())
                for (const auto d : enum_iterator1D<EmbeddingDimension>())
                    w_lm_head_cpu.set(t, d, static_cast<float16>(w_j.at(i++).template get<float>()));
            W_lm_head.copy_from_cpu(queue, w_lm_head_cpu);
        }
        V_lm_head.zero(queue);
        S_lm_head.zero(queue);
    }

    nlohmann::json OutputLayer::save() const
    {
        cpu_fixed_matrix<float16, TokenID, EmbeddingDimension> w_lm_head_cpu;
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        W_lm_head.copy_to_cpu(queue, w_lm_head_cpu);

        auto w_j = nlohmann::json::array();
        w_j.get_ref<nlohmann::json::array_t&>().reserve(W_lm_head.ROWS * W_lm_head.COLS);
        for (const auto t : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                w_j.push_back(static_cast<float>(w_lm_head_cpu.get(t, d)));
        return {{"W_lm_head", std::move(w_j)}};
    }


    bool TextTrainer::load(const std::string& filename)
    {
        m_optimizer_step = 0;
        m_has_loaded_training_state = false;
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
        {
            std::println("Unsupported model version in {}", filename);
            return false;
        }

        const auto expected_vocab_size = static_cast<size_t>(TokenID::MAX);
        if (j.contains("tokenizer_vocab_size"))
        {
            const auto model_vocab_size = j.at("tokenizer_vocab_size").template get<size_t>();
            if (model_vocab_size != expected_vocab_size)
            {
                std::println("Tokenizer vocab size mismatch in {}: model={}, runtime={}", filename, model_vocab_size, expected_vocab_size);
                return false;
            }
        }

        if (j.contains("tokenizer_signature"))
        {
            const auto model_sig = j.at("tokenizer_signature").template get<uint64_t>();
            const auto runtime_sig = tokenizer_signature();
            if (model_sig != runtime_sig)
            {
                std::println("Tokenizer signature mismatch in {}: rebuilding/retraining required", filename);
                return false;
            }
        }

        m_input_layer.load(j.at("input_layer"));

        if (j.contains("transformer_blocks"))
        {
            const auto& blocks_j = j.at("transformer_blocks");
            const size_t requested_blocks = m_transformer_blocks.size();
            const size_t saved_blocks = blocks_j.size();
            const size_t target_blocks = std::max(requested_blocks, saved_blocks);
            m_transformer_blocks.clear();
            m_transformer_blocks.reserve(target_blocks);
            for (const auto& b : blocks_j)
            {
                m_transformer_blocks.emplace_back();
                m_transformer_blocks.back().load(b);
            }
            for (size_t i = saved_blocks; i < target_blocks; ++i)
            {
                auto& block = m_transformer_blocks.emplace_back();
                block.randomize();
            }
            if (target_blocks != saved_blocks)
            {
                std::println(
                    "Extended model from {} to {} transformer block(s) while loading {}",
                    saved_blocks,
                    target_blocks,
                    filename
                );
            }
            reset_workspaces();
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

    void TextTrainer::save(const std::string& filename) const
    {
        if (is_safetensors_model_filename(filename))
        {
            save_to_safetensors(filename);
            if (!m_training_parameters_json.empty())
            {
                const auto parameters_filename = filename + ".training.json";
                std::ofstream parameters_file{parameters_filename};
                parameters_file << m_training_parameters_json << '\n';
                std::println("Saved training parameters '{}'", parameters_filename);
            }
            flush_training_progress_log();
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
        if (!m_training_parameters_json.empty())
        {
            const auto parameters_filename = filename + ".training.json";
            std::ofstream parameters_file{parameters_filename};
            parameters_file << m_training_parameters_json << '\n';
            std::println("Saved training parameters '{}'", parameters_filename);
        }
        flush_training_progress_log();
    }

    void TextTrainer::checkpoint() const
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
        void push_cpu_matrix(safetensors::safetensors_t& st, const std::string& key, const cpu_fixed_matrix<T, X, Y>& cpu_tmp, std::vector<uint8_t>& storage)
        {

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
        void push_matrix(safetensors::safetensors_t& st, const std::string& key, const fixed_size_matrix<T, X, Y>& m, std::vector<uint8_t>& storage)
        {
            cpu_fixed_matrix<T, X, Y> cpu_tmp;
            auto& queue = rllm::vulkan_runtime::get_queue(0);
            m.copy_to_cpu(queue, cpu_tmp);
            push_cpu_matrix(st, key, cpu_tmp, storage);
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
            auto& queue = rllm::vulkan_runtime::get_queue(0);
            m.copy_from_cpu(queue, cpu_tmp);
        }

        template <typename T, typename X, typename Y>
        void pull_cpu_matrix(const std::string& key, const safetensors::safetensors_t& st, cpu_fixed_matrix<T, X, Y>& m)
        {
            safetensors::tensor_t tensor;
            if (!st.tensors.at(key, &tensor) || tensor.dtype != safetensors::dtype::kFLOAT32)
            {
                std::println("Invalid or missing training tensor {}", key);
                std::abort();
            }
            const size_t rows = static_cast<size_t>(X::MAX);
            const size_t cols = static_cast<size_t>(Y::MAX);
            if (tensor.shape.size() != 2 || tensor.shape[0] != rows || tensor.shape[1] != cols)
            {
                std::println("Shape mismatch for training tensor {}", key);
                std::abort();
            }
            const uint8_t* base = st.storage.empty() ? st.databuffer_addr : reinterpret_cast<const uint8_t*>(st.storage.data());
            const auto* data = reinterpret_cast<float const*>(base + tensor.data_offsets[0]);
            for (size_t x = 0; x < rows; ++x)
                for (size_t y = 0; y < cols; ++y)
                    m.set(static_cast<X>(x), static_cast<Y>(y), static_cast<T>(data[x * cols + y]));
        }

    } // anonymous namespace


    // ---- InputLayer -----------------------------------------------------------

    void InputLayer::load_from_safetensors(const std::string& filename, std::string* err)
    {
        m_adam_first.zero();
        m_adam_second.zero();
        safetensors::safetensors_t st;
        if (!safetensors::load_from_file(filename, &st, nullptr, err))
        {
            if (err)
                *err = "Failed to read safetensors: " + filename;
            return;
        }

        pull_matrix("input_layer.embeddings", st, m_embeddings);
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_embeddings.copy_to_cpu(queue, m_embeddings_cpu);
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

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        V_q.zero(queue);
        V_k.zero(queue);
        V_v.zero(queue);
        V_o.zero(queue);
        V_gate.zero(queue);
        V_up.zero(queue);
        V_down.zero(queue);
        S_q.zero(queue);
        S_k.zero(queue);
        S_v.zero(queue);
        S_o.zero(queue);
        S_gate.zero(queue);
        S_up.zero(queue);
        S_down.zero(queue);
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_inputs.zero(queue);
        V_lm_head.zero(queue);
        S_lm_head.zero(queue);
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


    // ---- TextTrainer-level save/load for checkpointing ----------------------

    void TextTrainer::save_to_safetensors(const std::string& filename) const
    {
        safetensors::safetensors_t st = make_safetensors_metadata();
        std::vector<uint8_t> storage;

        // Input layer (friend grants access to m_embeddings)
        push_matrix(st, "input_layer.embeddings", m_input_layer.m_embeddings, storage);
        push_cpu_matrix(st, "training.input_layer.adam_first", m_input_layer.m_adam_first, storage);
        push_cpu_matrix(st, "training.input_layer.adam_second", m_input_layer.m_adam_second, storage);

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
            push_matrix(st, "training." + pfx + "V_q", block.V_q, storage);
            push_matrix(st, "training." + pfx + "V_k", block.V_k, storage);
            push_matrix(st, "training." + pfx + "V_v", block.V_v, storage);
            push_matrix(st, "training." + pfx + "V_o", block.V_o, storage);
            push_matrix(st, "training." + pfx + "V_gate", block.V_gate, storage);
            push_matrix(st, "training." + pfx + "V_up", block.V_up, storage);
            push_matrix(st, "training." + pfx + "V_down", block.V_down, storage);
            push_matrix(st, "training." + pfx + "S_q", block.S_q, storage);
            push_matrix(st, "training." + pfx + "S_k", block.S_k, storage);
            push_matrix(st, "training." + pfx + "S_v", block.S_v, storage);
            push_matrix(st, "training." + pfx + "S_o", block.S_o, storage);
            push_matrix(st, "training." + pfx + "S_gate", block.S_gate, storage);
            push_matrix(st, "training." + pfx + "S_up", block.S_up, storage);
            push_matrix(st, "training." + pfx + "S_down", block.S_down, storage);
        }

        // Output layers (friend grants access to m_output_layers internals)
        for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
        {
            const auto out_key = "output_layers." + std::to_string(static_cast<size_t>(oi)) + ".W_lm_head";
            push_matrix(st, out_key, m_output_layers[oi].W_lm_head, storage);
            const auto training_prefix = "training.output_layers." + std::to_string(static_cast<size_t>(oi)) + ".";
            push_matrix(st, training_prefix + "V_lm_head", m_output_layers[oi].V_lm_head, storage);
            push_matrix(st, training_prefix + "S_lm_head", m_output_layers[oi].S_lm_head, storage);
        }

        // Tokenizer metadata for validation on load.
        st.metadata.insert("tokenizer_vocab_size", std::to_string(static_cast<size_t>(TokenID::MAX)));
        st.metadata.insert("tokenizer_signature", std::to_string(tokenizer_signature()));
        if (m_learning_rate_provider)
        {
            st.metadata.insert("training_state_version", "1");
            st.metadata.insert("training.optimizer_step", std::to_string(m_optimizer_step));
            st.metadata.insert("training.learning_rate", std::to_string(m_learning_rate));
            st.metadata.insert("training.layer_learning_rate_multiplier", std::to_string(m_layer_learning_rate_multiplier));
            st.metadata.insert("training.learning_rate_schedule", std::to_string(static_cast<int>(m_learning_rate_schedule)));
            st.metadata.insert("training.learning_rate_schedule_steps", std::to_string(m_learning_rate_schedule_steps));
            st.metadata.insert("training.simulated_annealing_decay_factor", std::to_string(m_simulated_annealing_decay_factor));
            st.metadata.insert("training.simulated_annealing_initial_multiplier", std::to_string(m_simulated_annealing_initial_multiplier));
            st.metadata.insert("training.simulated_annealing_decay_epochs", std::to_string(m_simulated_annealing_decay_epochs));
            st.metadata.insert("training.simulated_annealing_min_multiplier", std::to_string(m_simulated_annealing_min_multiplier));
            st.metadata.insert("training.learning_rate_step", std::to_string(m_learning_rate_provider->step()));
            st.metadata.insert("training.current_learning_rate", std::to_string(m_learning_rate_provider->current_rate()));
            st.metadata.insert("training.epochs_at_current_rate", std::to_string(m_learning_rate_provider->epochs_at_current_rate()));
            st.metadata.insert("training.epoch", std::to_string(m_checkpoint_epoch));
            st.metadata.insert("training.window", std::to_string(m_checkpoint_window));
            st.metadata.insert("training.window_count", std::to_string(m_checkpoint_window_count));
            st.metadata.insert("training.rng_state", m_checkpoint_rng_state);
        }

        st.storage = std::move(storage);
        std::string warn;
        std::string err;
        const std::string tmp_filename = filename + ".tmp";
        if (!safetensors::save_to_file(st, tmp_filename, &warn, &err))
        {
            std::println("Failed to write safetensors file '{}': {}", filename, err);
            std::abort();
        }
        std::error_code ec;
        std::filesystem::rename(tmp_filename, filename, ec);
        if (ec)
        {
            std::filesystem::remove(filename, ec);
            ec.clear();
            std::filesystem::rename(tmp_filename, filename, ec);
        }
        if (ec)
        {
            std::println("Failed to replace safetensors file '{}': {}", filename, ec.message());
            std::filesystem::remove(tmp_filename);
            std::abort();
        }
    }

    bool TextTrainer::load_from_safetensors(const std::string& filename)
    {
        m_optimizer_step = 0;
        safetensors::safetensors_t st;
        std::string warn, ere;
        if (!safetensors::load_from_file(filename, &st, &warn, &ere))
        {
            std::println("Failed to read safetensors file: {}", filename);
            if (!ere.empty())
                std::println("safetensors error: {}", ere);
            return false;
        }

        const auto& header = st.metadata;
        if (!header.count("version"))
        {
            std::println("Missing version in safetensors metadata for: {}", filename);
            return false;
        }

        std::string training_state_version;
        if (header.count("training_state_version") &&
            header.at("training_state_version", &training_state_version) &&
            training_state_version == "1")
        {
            const auto parse_value = [&](const char* key, auto& value) {
                std::string text;
                return header.count(key) && header.at(key, &text) &&
                    parse_metadata_number(text, value);
            };
            const auto read_string = [&](const char* key, std::string& value) {
                return header.count(key) && header.at(key, &value);
            };
            int schedule = 0;
            if (!parse_value("training.optimizer_step", m_optimizer_step) ||
                !parse_value("training.learning_rate", m_learning_rate) ||
                !parse_value("training.layer_learning_rate_multiplier", m_layer_learning_rate_multiplier) ||
                !parse_value("training.learning_rate_schedule", schedule) ||
                !parse_value("training.learning_rate_schedule_steps", m_learning_rate_schedule_steps) ||
                !parse_value("training.simulated_annealing_decay_factor", m_simulated_annealing_decay_factor) ||
                !parse_value("training.simulated_annealing_initial_multiplier", m_simulated_annealing_initial_multiplier) ||
                !parse_value("training.simulated_annealing_decay_epochs", m_simulated_annealing_decay_epochs) ||
                !parse_value("training.simulated_annealing_min_multiplier", m_simulated_annealing_min_multiplier) ||
                !parse_value("training.learning_rate_step", m_loaded_learning_rate_step) ||
                !parse_value("training.current_learning_rate", m_loaded_current_learning_rate) ||
                !parse_value("training.epochs_at_current_rate", m_loaded_epochs_at_current_rate) ||
                !parse_value("training.epoch", m_checkpoint_epoch) ||
                !parse_value("training.window", m_checkpoint_window) ||
                !read_string("training.rng_state", m_checkpoint_rng_state) ||
                schedule < static_cast<int>(LearningRateSchedule::Constant) ||
                schedule > static_cast<int>(LearningRateSchedule::SimulatedAnnealing))
            {
                std::println("Invalid or incomplete training state in {}", filename);
                return false;
            }
            m_learning_rate_schedule = static_cast<LearningRateSchedule>(schedule);
            std::string window_count_text;
            if (header.count("training.window_count") && header.at("training.window_count", &window_count_text) &&
                !parse_metadata_number(window_count_text, m_checkpoint_window_count))
            {
                std::println("Invalid training window count in {}", filename);
                return false;
            }
            m_has_loaded_training_state = true;
        }

        const auto expected_vocab_size = static_cast<size_t>(TokenID::MAX);
        std::string vocab_size_str;
        if (header.count("tokenizer_vocab_size") && header.at("tokenizer_vocab_size", &vocab_size_str))
        {
            uint64_t model_vocab_size = 0;
            if (!parse_metadata_u64(vocab_size_str, model_vocab_size))
            {
                std::println("Invalid tokenizer vocab size metadata in {}: {}", filename, vocab_size_str);
                return false;
            }
            if (model_vocab_size != expected_vocab_size)
            {
                std::println(
                    "Tokenizer vocab size mismatch in {}: model={}, runtime={}",
                    filename,
                    model_vocab_size,
                    expected_vocab_size
                );
                return false;
            }
        }

        // Validate tokenizer if present (matches JSON load behavior).
        std::string sig_str;
        if (header.count("tokenizer_signature") && header.at("tokenizer_signature", &sig_str))
        {
            uint64_t model_sig = 0;
            if (!parse_metadata_u64(sig_str, model_sig))
            {
                std::println("Invalid tokenizer signature metadata in {}: {}", filename, sig_str);
                return false;
            }
            const auto runtime_sig = tokenizer_signature();
            if (model_sig != runtime_sig)
            {
                std::println("Tokenizer signature mismatch between safetensors model and runtime tokenizer for: {}", filename);
                return false;
            }
        }

        // Count transformer blocks by counting W_q tensors.
        size_t num_blocks = 0;
        while (st.tensors.count("transformer_blocks." + std::to_string(num_blocks) + ".W_q"))
            ++num_blocks;
        const size_t requested_blocks = m_transformer_blocks.size();
        const size_t target_blocks = std::max(requested_blocks, num_blocks);

        m_input_layer.load_from_safetensors(filename);
        if (m_has_loaded_training_state)
        {
            pull_cpu_matrix("training.input_layer.adam_first", st, m_input_layer.m_adam_first);
            pull_cpu_matrix("training.input_layer.adam_second", st, m_input_layer.m_adam_second);
        }

        // Clear existing blocks and create new ones.
        m_transformer_blocks.clear();
        m_transformer_blocks.reserve(target_blocks);

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
            if (m_has_loaded_training_state)
            {
                const std::string tpfx = "training." + pfx;
                pull_matrix(tpfx + "V_q", st, block.V_q);
                pull_matrix(tpfx + "V_k", st, block.V_k);
                pull_matrix(tpfx + "V_v", st, block.V_v);
                pull_matrix(tpfx + "V_o", st, block.V_o);
                pull_matrix(tpfx + "V_gate", st, block.V_gate);
                pull_matrix(tpfx + "V_up", st, block.V_up);
                pull_matrix(tpfx + "V_down", st, block.V_down);
                pull_matrix(tpfx + "S_q", st, block.S_q);
                pull_matrix(tpfx + "S_k", st, block.S_k);
                pull_matrix(tpfx + "S_v", st, block.S_v);
                pull_matrix(tpfx + "S_o", st, block.S_o);
                pull_matrix(tpfx + "S_gate", st, block.S_gate);
                pull_matrix(tpfx + "S_up", st, block.S_up);
                pull_matrix(tpfx + "S_down", st, block.S_down);
            }
        }
        for (size_t bi = num_blocks; bi < target_blocks; ++bi)
        {
            auto& block = m_transformer_blocks.emplace_back();
            block.randomize();
        }
        if (target_blocks != num_blocks)
        {
            std::println(
                "Extended model from {} to {} transformer block(s) while loading {}",
                num_blocks,
                target_blocks,
                filename
            );
        }
        reset_workspaces();

        if (st.tensors.count("output_layers.0.W_lm_head"))
        {
            for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
            {
                const auto out_key = "output_layers." + std::to_string(static_cast<size_t>(oi)) + ".W_lm_head";
                pull_matrix(out_key, st, m_output_layers[oi].W_lm_head);
                auto& queue = rllm::vulkan_runtime::get_queue(0);
                m_output_layers[oi].m_inputs.zero(queue);
                if (m_has_loaded_training_state)
                {
                    const auto tpfx = "training.output_layers." + std::to_string(static_cast<size_t>(oi)) + ".";
                    pull_matrix(tpfx + "V_lm_head", st, m_output_layers[oi].V_lm_head);
                    pull_matrix(tpfx + "S_lm_head", st, m_output_layers[oi].S_lm_head);
                }
                else
                {
                    m_output_layers[oi].V_lm_head.zero(queue);
                    m_output_layers[oi].S_lm_head.zero(queue);
                }
            }
        }
        else if (st.tensors.count("output_layers.W_lm_head"))
        {
            for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
                m_output_layers[oi].load_from_safetensors(filename);
        }

        return true;
    }

} // namespace rllm

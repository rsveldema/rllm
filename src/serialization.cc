#include <JsonTensorHelpers.hpp>
#include <ModelArtifacts.hpp>
#include <Trainer.hpp>
#include <safetensors.hh>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <print>
#include <string>
#include <string_view>


namespace rllm
{
    std::optional<size_t> upgrade_frozen_transformer_block_count(
        size_t saved_blocks, size_t requested_blocks, size_t saved_frozen_blocks,
        bool freeze_old_blocks)
    {
        const bool starting_upgrade =
            saved_blocks > 0 && requested_blocks > saved_blocks &&
            saved_frozen_blocks < saved_blocks;
        const bool resuming_upgrade =
            saved_blocks > 0 && requested_blocks == saved_blocks &&
            saved_frozen_blocks < saved_blocks;
        if (!starting_upgrade && !resuming_upgrade)
            return std::nullopt;
        if (starting_upgrade && freeze_old_blocks)
            return saved_blocks;
        return saved_frozen_blocks;
    }

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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_adam_first.zero(queue);
        m_adam_second.zero(queue);
        m_identifier_adam_first.zero(queue);
        m_identifier_adam_second.zero(queue);
        if (!j.contains("embeddings"))
            return;

        json_helpers::deserialize_matrix(j.at("embeddings"), m_embeddings_cpu);
        m_embeddings.copy_from_cpu(queue, m_embeddings_cpu);
        cpu_fixed_matrix<float16, IdentifierHashBucket, EmbeddingDimension> identifier_embeddings_cpu;
        json_helpers::deserialize_matrix(j.at("identifier_embeddings"), identifier_embeddings_cpu);
        m_identifier_embeddings.copy_from_cpu(queue, identifier_embeddings_cpu);
    }

    nlohmann::json InputLayer::save() const
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_embeddings.copy_to_cpu(queue, const_cast<cpu_fixed_matrix<float16, TokenID, EmbeddingDimension>&>(m_embeddings_cpu));
        cpu_fixed_matrix<float16, IdentifierHashBucket, EmbeddingDimension> identifier_embeddings_cpu;
        m_identifier_embeddings.copy_to_cpu(queue, identifier_embeddings_cpu);
        queue.wait("InputLayer JSON serialization");
        return {
            {"embeddings", *json_helpers::serialize_matrix(m_embeddings_cpu)},
            {"identifier_embeddings", *json_helpers::serialize_matrix(identifier_embeddings_cpu)}
        };
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
        if (j.contains("W_string_table_index_head"))
        {
            cpu_fixed_matrix<float16, PositionIndex, EmbeddingDimension> w_string_table_index_head_cpu;
            const auto& w_j = j.at("W_string_table_index_head");
            size_t i = 0;
            for (const auto t : enum_iterator1D<PositionIndex>())
                for (const auto d : enum_iterator1D<EmbeddingDimension>())
                    w_string_table_index_head_cpu.set(t, d, static_cast<float16>(w_j.at(i++).template get<float>()));
            W_string_table_index_head.copy_from_cpu(queue, w_string_table_index_head_cpu);
        }
        else
            W_string_table_index_head.zero(queue);
        V_lm_head.zero(queue);
        S_lm_head.zero(queue);
        V_string_table_index_head.zero(queue);
        S_string_table_index_head.zero(queue);
    }

    nlohmann::json OutputLayer::save() const
    {
        cpu_fixed_matrix<float16, TokenID, EmbeddingDimension> w_lm_head_cpu;
        cpu_fixed_matrix<float16, PositionIndex, EmbeddingDimension> w_string_table_index_head_cpu;
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        W_lm_head.copy_to_cpu(queue, w_lm_head_cpu);
        W_string_table_index_head.copy_to_cpu(queue, w_string_table_index_head_cpu);

        auto w_j = nlohmann::json::array();
        w_j.get_ref<nlohmann::json::array_t&>().reserve(W_lm_head.ROWS * W_lm_head.COLS);
        for (const auto t : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                w_j.push_back(static_cast<float>(w_lm_head_cpu.get(t, d)));
        auto w_string_table_index_j = nlohmann::json::array();
        w_string_table_index_j.get_ref<nlohmann::json::array_t&>().reserve(
            W_string_table_index_head.ROWS * W_string_table_index_head.COLS);
        for (const auto t : enum_iterator1D<PositionIndex>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                w_string_table_index_j.push_back(static_cast<float>(
                    w_string_table_index_head_cpu.get(t, d)));
        return {
            {"W_lm_head", std::move(w_j)},
            {"W_string_table_index_head", std::move(w_string_table_index_j)}
        };
    }


    bool TextTrainer::load(const std::string& filename)
    {
        m_optimizer_step = 0;
        m_has_loaded_training_state = false;
        m_loaded_model_path.clear();
        m_frozen_transformer_block_count = 0;
        if (is_safetensors_model_filename(filename))
        {
            const bool loaded = load_from_safetensors(filename);
            if (loaded)
                m_loaded_model_path = std::filesystem::absolute(filename).lexically_normal();
            return loaded;
        }

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
            const size_t saved_frozen_blocks = j.value("frozen_transformer_blocks", size_t{0});
            if (m_upgrade_mode)
            {
                const auto frozen_blocks = upgrade_frozen_transformer_block_count(
                    saved_blocks, requested_blocks, saved_frozen_blocks,
                    m_freeze_old_blocks_on_upgrade);
                if (!frozen_blocks)
                {
                    std::println(
                        "Upgrade mode requires a target deeper than its checkpoint, or a resumable checkpoint at the requested depth; got {} saved, {} requested, {} frozen",
                        saved_blocks, requested_blocks, saved_frozen_blocks);
                    return false;
                }
                m_frozen_transformer_block_count = *frozen_blocks;
            }
            else
            {
                if (saved_frozen_blocks > target_blocks)
                {
                    std::println("Invalid frozen transformer block count {} in {}", saved_frozen_blocks, filename);
                    return false;
                }
                m_frozen_transformer_block_count = saved_frozen_blocks;
            }
            if (m_all_blocks_read_write)
                m_frozen_transformer_block_count = 0;
            m_transformer_blocks.clear();
            m_transformer_blocks.reserve(target_blocks);
            for (const auto& b : blocks_j)
            {
                m_transformer_blocks.emplace_back();
                m_transformer_blocks.back().set_adam_second_moments_enabled(
                    m_training_algorithm == TrainingAlgorithm::Dust);
                m_transformer_blocks.back().load(b);
            }
            for (size_t i = saved_blocks; i < target_blocks; ++i)
            {
                auto& block = m_transformer_blocks.emplace_back();
                block.set_adam_second_moments_enabled(
                    m_training_algorithm == TrainingAlgorithm::Dust);
                block.randomize(
                    m_weight_initializer, m_ffn_initializer,
                    m_upgrade_residual_output_scale);
            }
            const size_t saved_loop_first = j.value("loop_first_block", size_t{0});
            const size_t saved_loop_last = j.value("loop_last_block", size_t{0});
            const size_t saved_loop_count = j.value("loop_count", size_t{1});
            const size_t saved_loop_architecture =
                j.value("loop_architecture_version", size_t{0});
            if (saved_loop_architecture > LOOP_ARCHITECTURE_VERSION)
            {
                std::println("Unsupported loop architecture version {} in {}",
                    saved_loop_architecture, filename);
                return false;
            }
            if (saved_loop_count == 0 ||
                (target_blocks != 0 &&
                 (saved_loop_first > saved_loop_last || saved_loop_last >= target_blocks)))
            {
                std::println("Invalid transformer loop configuration in {}", filename);
                return false;
            }
            if (!m_transformer_loop_override_on_load)
            {
                m_loop_first_block = saved_loop_first;
                m_loop_last_block = saved_loop_last;
                m_loop_count = saved_loop_count;
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
        m_loaded_model_path = std::filesystem::absolute(filename).lexically_normal();
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
        j["frozen_transformer_blocks"] = m_frozen_transformer_block_count;
        j["loop_first_block"] = m_loop_first_block;
        j["loop_last_block"] = m_loop_last_block;
        j["loop_count"] = m_loop_count;
        j["loop_architecture_version"] = LOOP_ARCHITECTURE_VERSION;
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

    std::filesystem::path TextTrainer::checkpoint() const
    {
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const auto filename = model_artifact_path(std::format("checkpoint-{}.st", now_ms));
        std::filesystem::create_directories(filename.parent_path());
        save(filename.string());
        return filename;
    }

    void TextTrainer::link_checkpoint_alias(
        const std::filesystem::path& checkpoint_filename,
        const std::filesystem::path& alias_filename) const
    {
        const auto replace_with_relative_symlink = [](
            const std::filesystem::path& target,
            const std::filesystem::path& alias) {
            const auto temporary = alias.parent_path() / ("." + alias.filename().string() + ".tmp");
            std::error_code error;
            std::filesystem::remove(temporary, error);
            error.clear();
            std::filesystem::create_symlink(target.filename(), temporary, error);
            if (error)
            {
                std::println(
                    "Failed to create checkpoint symlink '{}' -> '{}': {}",
                    temporary.string(), target.filename().string(), error.message());
                std::exit(1);
            }
            std::filesystem::rename(temporary, alias, error);
            if (error)
            {
                std::filesystem::remove(temporary);
                std::println(
                    "Failed to publish checkpoint symlink '{}' as '{}': {}",
                    temporary.string(), alias.string(), error.message());
                std::exit(1);
            }
        };

        replace_with_relative_symlink(checkpoint_filename, alias_filename);
        const std::filesystem::path checkpoint_sidecar = checkpoint_filename.string() + ".training.json";
        const std::filesystem::path alias_sidecar = alias_filename.string() + ".training.json";
        if (std::filesystem::is_regular_file(checkpoint_sidecar))
            replace_with_relative_symlink(checkpoint_sidecar, alias_sidecar);
        else
            std::filesystem::remove(alias_sidecar);
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_adam_first.zero(queue);
        m_adam_second.zero(queue);
        m_identifier_adam_first.zero(queue);
        m_identifier_adam_second.zero(queue);
        safetensors::safetensors_t st;
        if (!safetensors::load_from_file(filename, &st, nullptr, err))
        {
            if (err)
                *err = "Failed to read safetensors: " + filename;
            return;
        }

        pull_matrix("input_layer.embeddings", st, m_embeddings);
        pull_matrix("input_layer.identifier_embeddings", st, m_identifier_embeddings);
        m_embeddings.copy_to_cpu(queue, m_embeddings_cpu);
    }

    void InputLayer::save_to_safetensors(const std::string& filename, std::string* warn, std::string* err) const
    {
        (void) warn;
        safetensors::safetensors_t st = make_safetensors_metadata();
        std::vector<uint8_t> storage;

        push_matrix(st, "input_layer.embeddings", m_embeddings, storage);
        push_matrix(st, "input_layer.identifier_embeddings", m_identifier_embeddings, storage);
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
        if (m_adam_second_moments)
        {
            m_adam_second_moments->q.zero(queue);
            m_adam_second_moments->k.zero(queue);
            m_adam_second_moments->v.zero(queue);
            m_adam_second_moments->o.zero(queue);
            m_adam_second_moments->gate.zero(queue);
            m_adam_second_moments->up.zero(queue);
            m_adam_second_moments->down.zero(queue);
        }
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
        pull_matrix("output_layers.W_string_table_index_head", st, W_string_table_index_head);
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_inputs.zero(queue);
        m_string_table_index_inputs.zero(queue);
        V_lm_head.zero(queue);
        S_lm_head.zero(queue);
        V_string_table_index_head.zero(queue);
        S_string_table_index_head.zero(queue);
    }

    void OutputLayer::save_to_safetensors(const std::string& filename, std::string* warn, std::string* err) const
    {
        (void) warn;
        safetensors::safetensors_t st = make_safetensors_metadata();
        std::vector<uint8_t> storage;

        push_matrix(st, "output_layers.W_lm_head", W_lm_head, storage);
        push_matrix(st, "output_layers.W_string_table_index_head", W_string_table_index_head, storage);
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
        push_matrix(st, "input_layer.identifier_embeddings", m_input_layer.m_identifier_embeddings, storage);
        push_matrix(st, "training.input_layer.adam_first", m_input_layer.m_adam_first, storage);
        push_matrix(st, "training.input_layer.adam_second", m_input_layer.m_adam_second, storage);
        push_matrix(st, "training.input_layer.identifier_adam_first", m_input_layer.m_identifier_adam_first, storage);
        push_matrix(st, "training.input_layer.identifier_adam_second", m_input_layer.m_identifier_adam_second, storage);

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
            if (m_training_algorithm == TrainingAlgorithm::Backprop)
            {
                push_matrix(st, "training." + pfx + "muon_momentum_q", block.V_q, storage);
                push_matrix(st, "training." + pfx + "muon_momentum_k", block.V_k, storage);
                push_matrix(st, "training." + pfx + "muon_momentum_v", block.V_v, storage);
                push_matrix(st, "training." + pfx + "muon_momentum_o", block.V_o, storage);
                push_matrix(st, "training." + pfx + "muon_momentum_gate", block.V_gate, storage);
                push_matrix(st, "training." + pfx + "muon_momentum_up", block.V_up, storage);
                push_matrix(st, "training." + pfx + "muon_momentum_down", block.V_down, storage);
            }
            else
            {
                assert(block.m_adam_second_moments);
                push_matrix(st, "training." + pfx + "V_q", block.V_q, storage);
                push_matrix(st, "training." + pfx + "V_k", block.V_k, storage);
                push_matrix(st, "training." + pfx + "V_v", block.V_v, storage);
                push_matrix(st, "training." + pfx + "V_o", block.V_o, storage);
                push_matrix(st, "training." + pfx + "V_gate", block.V_gate, storage);
                push_matrix(st, "training." + pfx + "V_up", block.V_up, storage);
                push_matrix(st, "training." + pfx + "V_down", block.V_down, storage);
                push_matrix(st, "training." + pfx + "S_q", block.m_adam_second_moments->q, storage);
                push_matrix(st, "training." + pfx + "S_k", block.m_adam_second_moments->k, storage);
                push_matrix(st, "training." + pfx + "S_v", block.m_adam_second_moments->v, storage);
                push_matrix(st, "training." + pfx + "S_o", block.m_adam_second_moments->o, storage);
                push_matrix(st, "training." + pfx + "S_gate", block.m_adam_second_moments->gate, storage);
                push_matrix(st, "training." + pfx + "S_up", block.m_adam_second_moments->up, storage);
                push_matrix(st, "training." + pfx + "S_down", block.m_adam_second_moments->down, storage);
            }
        }

        // Output layers (friend grants access to m_output_layers internals)
        for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
        {
            const auto out_key = "output_layers." + std::to_string(static_cast<size_t>(oi)) + ".W_lm_head";
            push_matrix(st, out_key, m_output_layers[oi].W_lm_head, storage);
            const auto out_index_key =
                "output_layers." + std::to_string(static_cast<size_t>(oi)) + ".W_string_table_index_head";
            push_matrix(st, out_index_key, m_output_layers[oi].W_string_table_index_head, storage);
            const auto training_prefix = "training.output_layers." + std::to_string(static_cast<size_t>(oi)) + ".";
            push_matrix(st, training_prefix + "V_lm_head", m_output_layers[oi].V_lm_head, storage);
            push_matrix(st, training_prefix + "S_lm_head", m_output_layers[oi].S_lm_head, storage);
            push_matrix(st, training_prefix + "V_string_table_index_head",
                m_output_layers[oi].V_string_table_index_head, storage);
            push_matrix(st, training_prefix + "S_string_table_index_head",
                m_output_layers[oi].S_string_table_index_head, storage);
        }

        // Tokenizer metadata for validation on load.
        st.metadata.insert("tokenizer_vocab_size", std::to_string(static_cast<size_t>(TokenID::MAX)));
        st.metadata.insert("tokenizer_signature", std::to_string(tokenizer_signature()));
        st.metadata.insert("frozen_transformer_blocks", std::to_string(m_frozen_transformer_block_count));
        st.metadata.insert("loop_first_block", std::to_string(m_loop_first_block));
        st.metadata.insert("loop_last_block", std::to_string(m_loop_last_block));
        st.metadata.insert("loop_count", std::to_string(m_loop_count));
        st.metadata.insert("loop_architecture_version", std::to_string(LOOP_ARCHITECTURE_VERSION));
        if (m_learning_rate_provider)
        {
            st.metadata.insert("training_state_version", "2");
            st.metadata.insert("training.optimizer",
                m_training_algorithm == TrainingAlgorithm::Backprop ? "muon+adamw" : "adamw");
            st.metadata.insert("training.optimizer_step", std::to_string(m_optimizer_step));
            st.metadata.insert("training.learning_rate", std::to_string(m_learning_rate));
            st.metadata.insert("training.layer_learning_rate_multiplier", std::to_string(m_layer_learning_rate_multiplier));
            st.metadata.insert("training.warmup_percent", std::to_string(m_warmup_percent));
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
            training_state_version == "2")
        {
            const auto parse_value = [&](const char* key, auto& value) {
                std::string text;
                return header.count(key) && header.at(key, &text) &&
                    parse_metadata_number(text, value);
            };
            const auto read_string = [&](const char* key, std::string& value) {
                return header.count(key) && header.at(key, &value);
            };
            std::string checkpoint_optimizer;
            const std::string expected_optimizer =
                m_training_algorithm == TrainingAlgorithm::Backprop ? "muon+adamw" : "adamw";
            if (!read_string("training.optimizer", checkpoint_optimizer) ||
                checkpoint_optimizer != expected_optimizer)
            {
                std::println("Checkpoint optimizer is incompatible with runtime backend in {}", filename);
                return false;
            }
            int schedule = 0;
            float checkpoint_learning_rate = 0.0f;
            float checkpoint_layer_learning_rate_multiplier = 0.0f;
            float checkpoint_simulated_annealing_decay_factor = 0.0f;
            float checkpoint_simulated_annealing_initial_multiplier = 0.0f;
            size_t checkpoint_simulated_annealing_decay_epochs = 0;
            float checkpoint_simulated_annealing_min_multiplier = 0.0f;
            std::string warmup_percent_text;
            float checkpoint_warmup_percent = 0.0f;
            if (header.count("training.warmup_percent") &&
                (!header.at("training.warmup_percent", &warmup_percent_text) ||
                 !parse_metadata_number(warmup_percent_text, checkpoint_warmup_percent) ||
                 !std::isfinite(checkpoint_warmup_percent) ||
                 checkpoint_warmup_percent <= 0.0f || checkpoint_warmup_percent > 100.0f))
            {
                std::println("Invalid warmup percentage in {}", filename);
                return false;
            }
            if (!parse_value("training.optimizer_step", m_optimizer_step) ||
                !parse_value("training.learning_rate", checkpoint_learning_rate) ||
                !parse_value("training.layer_learning_rate_multiplier", checkpoint_layer_learning_rate_multiplier) ||
                !parse_value("training.learning_rate_schedule", schedule) ||
                !parse_value("training.learning_rate_schedule_steps", m_learning_rate_schedule_steps) ||
                !parse_value("training.simulated_annealing_decay_factor", checkpoint_simulated_annealing_decay_factor) ||
                !parse_value("training.simulated_annealing_initial_multiplier", checkpoint_simulated_annealing_initial_multiplier) ||
                !parse_value("training.simulated_annealing_decay_epochs", checkpoint_simulated_annealing_decay_epochs) ||
                !parse_value("training.simulated_annealing_min_multiplier", checkpoint_simulated_annealing_min_multiplier) ||
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
        const bool resized_model = target_blocks != num_blocks;
        size_t saved_frozen_blocks = 0;
        std::string frozen_blocks_text;
        if (header.count("frozen_transformer_blocks") &&
            (!header.at("frozen_transformer_blocks", &frozen_blocks_text) ||
             !parse_metadata_number(frozen_blocks_text, saved_frozen_blocks)))
        {
            std::println("Invalid frozen transformer block count in {}", filename);
            return false;
        }
        if (m_upgrade_mode)
        {
            const auto frozen_blocks = upgrade_frozen_transformer_block_count(
                num_blocks, requested_blocks, saved_frozen_blocks,
                m_freeze_old_blocks_on_upgrade);
            if (!frozen_blocks)
            {
                std::println(
                    "Upgrade mode requires a target deeper than its checkpoint, or a resumable checkpoint at the requested depth; got {} saved, {} requested, {} frozen",
                    num_blocks, requested_blocks, saved_frozen_blocks);
                return false;
            }
            m_frozen_transformer_block_count = *frozen_blocks;
        }
        else
        {
            if (saved_frozen_blocks > target_blocks)
            {
                std::println("Invalid frozen transformer block count {} in {}", saved_frozen_blocks, filename);
                return false;
            }
            m_frozen_transformer_block_count = saved_frozen_blocks;
        }
        if (m_all_blocks_read_write)
            m_frozen_transformer_block_count = 0;

        size_t saved_loop_first = 0;
        size_t saved_loop_last = 0;
        size_t saved_loop_count = 1;
        size_t saved_loop_architecture = 0;
        const auto parse_optional_loop_metadata = [&](const char* name, size_t& value) {
            std::string text;
            return !header.count(name) ||
                (header.at(name, &text) && parse_metadata_number(text, value));
        };
        if (!parse_optional_loop_metadata("loop_first_block", saved_loop_first) ||
            !parse_optional_loop_metadata("loop_last_block", saved_loop_last) ||
            !parse_optional_loop_metadata("loop_count", saved_loop_count) ||
            !parse_optional_loop_metadata("loop_architecture_version", saved_loop_architecture) ||
            saved_loop_architecture > LOOP_ARCHITECTURE_VERSION ||
            saved_loop_count == 0 ||
            (target_blocks != 0 &&
             (saved_loop_first > saved_loop_last || saved_loop_last >= target_blocks)))
        {
            std::println("Invalid transformer loop configuration in {}", filename);
            return false;
        }
        const bool loop_configuration_changed = !m_transformer_loop_override_on_load &&
            (m_loop_first_block != saved_loop_first ||
             m_loop_last_block != saved_loop_last ||
             m_loop_count != saved_loop_count);
        if (!m_transformer_loop_override_on_load)
        {
            m_loop_first_block = saved_loop_first;
            m_loop_last_block = saved_loop_last;
            m_loop_count = saved_loop_count;
        }

        if ((resized_model || m_reset_training_cursor_on_load) && m_has_loaded_training_state)
        {
            m_checkpoint_epoch = 0;
            m_checkpoint_window = 0;
            m_checkpoint_window_count = 0;
            m_reset_optimizer_state_on_load = true;
            m_restart_learning_rate_schedule_on_load = true;
            m_learning_rate_schedule_steps = 0;
            if (resized_model)
                std::println(
                    "Reset training cursor, optimizer, and learning-rate schedule after extending model from {} to {} transformer blocks",
                    num_blocks,
                    target_blocks);
            else
                std::println(
                    "Reset training cursor, optimizer, and learning-rate schedule as explicitly requested");
        }

        const bool load_optimizer_tensors =
            m_has_loaded_training_state &&
            m_optimizer_step != 0 &&
            !m_reset_optimizer_state_on_load;
        if (m_has_loaded_training_state && !load_optimizer_tensors)
        {
            LOG_INFO(
                "Initializing Adam moments on the device instead of uploading checkpoint tensors ({})",
                m_reset_optimizer_state_on_load ? "optimizer reset requested" : "optimizer step is zero");
        }

        m_input_layer.load_from_safetensors(filename);
        if (load_optimizer_tensors)
        {
            pull_matrix("training.input_layer.adam_first", st, m_input_layer.m_adam_first);
            pull_matrix("training.input_layer.adam_second", st, m_input_layer.m_adam_second);
            pull_matrix("training.input_layer.identifier_adam_first", st, m_input_layer.m_identifier_adam_first);
            pull_matrix("training.input_layer.identifier_adam_second", st, m_input_layer.m_identifier_adam_second);
        }

        // Same-depth resume already has correctly sized device allocations.
        // Reusing them avoids fragmenting a nearly full device heap and avoids
        // transiently rebuilding the large forward/backward workspaces.
        const bool rebuild_blocks = m_transformer_blocks.size() != target_blocks;
        if (rebuild_blocks)
        {
            m_transformer_blocks.clear();
            m_transformer_blocks.reserve(target_blocks);
            for (size_t bi = 0; bi < target_blocks; ++bi)
            {
                auto& block = m_transformer_blocks.emplace_back();
                block.set_adam_second_moments_enabled(
                    m_training_algorithm == TrainingAlgorithm::Dust);
            }
        }

        for (size_t bi = 0; bi < num_blocks; ++bi)
        {
            std::string pfx = "transformer_blocks." + std::to_string(bi) + ".";

            auto& block = m_transformer_blocks[bi];
            auto& queue = rllm::vulkan_runtime::get_queue(0);

            // Directly pull each weight matrix into the block's members.
            pull_matrix(pfx + "W_q", st, block.W_q);
            pull_matrix(pfx + "W_k", st, block.W_k);
            pull_matrix(pfx + "W_v", st, block.W_v);
            pull_matrix(pfx + "W_o", st, block.W_o);
            pull_matrix(pfx + "W_gate", st, block.W_gate);
            pull_matrix(pfx + "W_up", st, block.W_up);
            pull_matrix(pfx + "W_down", st, block.W_down);
            if (load_optimizer_tensors)
            {
                const std::string tpfx = "training." + pfx;
                if (m_training_algorithm == TrainingAlgorithm::Backprop)
                {
                    pull_matrix(tpfx + "muon_momentum_q", st, block.V_q);
                    pull_matrix(tpfx + "muon_momentum_k", st, block.V_k);
                    pull_matrix(tpfx + "muon_momentum_v", st, block.V_v);
                    pull_matrix(tpfx + "muon_momentum_o", st, block.V_o);
                    pull_matrix(tpfx + "muon_momentum_gate", st, block.V_gate);
                    pull_matrix(tpfx + "muon_momentum_up", st, block.V_up);
                    pull_matrix(tpfx + "muon_momentum_down", st, block.V_down);
                }
                else
                {
                    assert(block.m_adam_second_moments);
                    pull_matrix(tpfx + "V_q", st, block.V_q);
                    pull_matrix(tpfx + "V_k", st, block.V_k);
                    pull_matrix(tpfx + "V_v", st, block.V_v);
                    pull_matrix(tpfx + "V_o", st, block.V_o);
                    pull_matrix(tpfx + "V_gate", st, block.V_gate);
                    pull_matrix(tpfx + "V_up", st, block.V_up);
                    pull_matrix(tpfx + "V_down", st, block.V_down);
                    pull_matrix(tpfx + "S_q", st, block.m_adam_second_moments->q);
                    pull_matrix(tpfx + "S_k", st, block.m_adam_second_moments->k);
                    pull_matrix(tpfx + "S_v", st, block.m_adam_second_moments->v);
                    pull_matrix(tpfx + "S_o", st, block.m_adam_second_moments->o);
                    pull_matrix(tpfx + "S_gate", st, block.m_adam_second_moments->gate);
                    pull_matrix(tpfx + "S_up", st, block.m_adam_second_moments->up);
                    pull_matrix(tpfx + "S_down", st, block.m_adam_second_moments->down);
                }
            }
            else
            {
                block.V_q.zero(queue);
                block.V_k.zero(queue);
                block.V_v.zero(queue);
                block.V_o.zero(queue);
                block.V_gate.zero(queue);
                block.V_up.zero(queue);
                block.V_down.zero(queue);
                if (block.m_adam_second_moments)
                {
                    block.m_adam_second_moments->q.zero(queue);
                    block.m_adam_second_moments->k.zero(queue);
                    block.m_adam_second_moments->v.zero(queue);
                    block.m_adam_second_moments->o.zero(queue);
                    block.m_adam_second_moments->gate.zero(queue);
                    block.m_adam_second_moments->up.zero(queue);
                    block.m_adam_second_moments->down.zero(queue);
                }
            }
        }
        for (size_t bi = num_blocks; bi < target_blocks; ++bi)
        {
            auto& block = m_transformer_blocks[bi];
            block.randomize(
                m_weight_initializer, m_ffn_initializer,
                m_upgrade_residual_output_scale);
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
        if (rebuild_blocks || loop_configuration_changed)
            reset_workspaces();

        if (st.tensors.count("output_layers.0.W_lm_head"))
        {
            for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
            {
                const auto out_key = "output_layers." + std::to_string(static_cast<size_t>(oi)) + ".W_lm_head";
                pull_matrix(out_key, st, m_output_layers[oi].W_lm_head);
                const auto out_index_key =
                    "output_layers." + std::to_string(static_cast<size_t>(oi)) + ".W_string_table_index_head";
                pull_matrix(out_index_key, st, m_output_layers[oi].W_string_table_index_head);
                auto& queue = rllm::vulkan_runtime::get_queue(0);
                m_output_layers[oi].m_inputs.zero(queue);
                m_output_layers[oi].m_string_table_index_inputs.zero(queue);
                if (load_optimizer_tensors)
                {
                    const auto tpfx = "training.output_layers." + std::to_string(static_cast<size_t>(oi)) + ".";
                    pull_matrix(tpfx + "V_lm_head", st, m_output_layers[oi].V_lm_head);
                    pull_matrix(tpfx + "S_lm_head", st, m_output_layers[oi].S_lm_head);
                    pull_matrix(tpfx + "V_string_table_index_head", st, m_output_layers[oi].V_string_table_index_head);
                    pull_matrix(tpfx + "S_string_table_index_head", st, m_output_layers[oi].S_string_table_index_head);
                }
                else
                {
                    m_output_layers[oi].V_lm_head.zero(queue);
                    m_output_layers[oi].S_lm_head.zero(queue);
                    m_output_layers[oi].V_string_table_index_head.zero(queue);
                    m_output_layers[oi].S_string_table_index_head.zero(queue);
                }
            }
        }

        return true;
    }

} // namespace rllm

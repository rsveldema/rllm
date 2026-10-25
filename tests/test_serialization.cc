// Test serialization of layers directly without needing Corpus or full NN setup.
#include <InputLayer.hpp>
#include <OutputLayer.hpp>
#include <TransformerBlock.hpp>
#include <Corpus.hpp>
#include <Statistics.hpp>
#include <TextTrainer.hpp>
#include <safetensors.hh>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

TEST(SerializationTest, UpgradeModeAcceptsAnyDepthIncreaseAndItsResume)
{
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(2, 4, 0, true), 2u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(2, 4, 0, false), 0u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(4, 4, 2, false), 2u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(4, 4, 0, true), 0u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(4, 6, 0, true), 4u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(4, 6, 0, false), 0u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(4, 6, 2, false), 2u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(8, 16, 0, true), 8u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(8, 16, 0, false), 0u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(16, 16, 8, false), 8u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(16, 16, 0, false), 0u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(16, 16, 0, true), 0u);

    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(8, 8, 0, true), 0u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(4, 8, 0, true), 4u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(2, 4, 1, true), 2u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(3, 3, 0, true), 0u);
    EXPECT_EQ(rllm::upgrade_frozen_transformer_block_count(16, 16, 4, false), 4u);
    EXPECT_FALSE(rllm::upgrade_frozen_transformer_block_count(6, 4, 0, true).has_value());
    EXPECT_FALSE(rllm::upgrade_frozen_transformer_block_count(4, 6, 4, true).has_value());
}

// ---------------------------------------------------------------------------
// JSON layer round-trip: InputLayer embeddings
// ---------------------------------------------------------------------------

TEST(SerializationTest, InputLayerJsonRoundTrip)
{
    rllm::InputLayer layer;
    layer.set_random_embeddings();

    auto j = layer.save();
    std::string json_str = j.dump(2);
    ASSERT_GT(json_str.size(), 0u);

    rllm::InputLayer loaded;
    loaded.load(nlohmann::json::parse(json_str));

    // Includes both category-token and hashed identifier-name embeddings.
    EXPECT_EQ(layer.save().dump(), loaded.save().dump());

    for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>()) {
        rllm::embedding_row_t a_emb;
        rllm::embedding_row_t b_emb;
        layer.get_embedding(tok, a_emb);
        loaded.get_embedding(tok, b_emb);
        for (const auto d : rllm::enum_iterator1D<rllm::EmbeddingDimension>()) {
            float a = a_emb[static_cast<size_t>(d)];
            float b = b_emb[static_cast<size_t>(d)];
            if (std::abs(a - b) > 1e-7f) ADD_FAILURE() << "embedding mismatch tok=" << static_cast<size_t>(tok);
        }
    }
}

// ---------------------------------------------------------------------------
// JSON layer round-trip: OutputLayer weights
// ---------------------------------------------------------------------------

TEST(SerializationTest, OutputLayerJsonRoundTrip)
{
    rllm::OutputLayer layer;
    auto j = layer.save();
    std::string json_str = j.dump(2);
    ASSERT_GT(json_str.size(), 0u);

    rllm::OutputLayer loaded;
    loaded.load(nlohmann::json::parse(json_str));

    // Compare via JSON serialization to avoid private member access.
    if (layer.save().dump() != loaded.save().dump()) ADD_FAILURE() << "json mismatch";
}

// ---------------------------------------------------------------------------
// JSON layer round-trip: TransformerBlock weights
// ---------------------------------------------------------------------------

TEST(SerializationTest, TransformerBlockJsonRoundTrip)
{
    rllm::TransformerBlock block;
    block.randomize();

    auto j = block.save();
    std::string json_str = j->dump(2);
    ASSERT_GT(json_str.size(), 0u);

    rllm::TransformerBlock loaded;
    loaded.load(nlohmann::json::parse(json_str));

    // Compare via JSON serialization to avoid private member access.
    auto original_json = block.save();
    auto loaded_json = loaded.save();
    std::string original_str = original_json->dump();
    std::string loaded_str = loaded_json->dump();
    if (original_str != loaded_str) ADD_FAILURE() << "json mismatch";
}

// ---------------------------------------------------------------------------
// Safetensors layer round-trip: InputLayer
// ---------------------------------------------------------------------------

TEST(SerializationTest, InputLayerSafetensorsRoundTrip)
{
    rllm::InputLayer layer;
    layer.set_random_embeddings();
    std::puts("InputLayer safetensors: step 1 - embeddings set");

    const std::string sf_file = (std::filesystem::temp_directory_path() / "input_layer.safetensors").string();
    std::string warn, err;
    std::puts("InputLayer safetensors: step 2 - calling save_to_safetensors...");
    layer.save_to_safetensors(sf_file, &warn, &err);
    std::printf("save returned. err=%s\n", (err.empty() ? "empty" : err.c_str()));
    EXPECT_TRUE(err.empty()) << "Safetensors save error: " << (err.empty() ? "none" : err);

    // Load back via the helper method.
    std::puts("step 3 - loading...");
    rllm::InputLayer loaded;
    std::string load_err;
    loaded.load_from_safetensors(sf_file, &load_err);
    EXPECT_TRUE(load_err.empty()) << "Safetensors load error: " << (load_err.empty() ? "none" : load_err);

    // Re-serializing compares the identifier-name table as well as token embeddings.
    EXPECT_EQ(layer.save().dump(), loaded.save().dump());

    // Verify embeddings match via public get_embedding API.
    for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>()) {
        rllm::embedding_row_t a_emb;
        rllm::embedding_row_t b_emb;
        layer.get_embedding(tok, a_emb);
        loaded.get_embedding(tok, b_emb);
        for (const auto d : rllm::enum_iterator1D<rllm::EmbeddingDimension>()) {
            float a = a_emb[static_cast<size_t>(d)];
            float b = b_emb[static_cast<size_t>(d)];
            if (std::abs(a - b) > 1e-7f)
                ADD_FAILURE() << "embedding mismatch tok=" << static_cast<size_t>(tok);
        }
    }

    std::filesystem::remove(sf_file);
}

// ---------------------------------------------------------------------------
// Safetensors layer round-trip: OutputLayer
// ---------------------------------------------------------------------------

TEST(SerializationTest, OutputLayerSafetensorsRoundTrip)
{
    rllm::OutputLayer layer;
    const std::string sf_file = (std::filesystem::temp_directory_path() / "output_layer.safetensors").string();
    std::string warn, err;
    layer.save_to_safetensors(sf_file, &warn, &err);
    EXPECT_TRUE(err.empty()) << "Safetensors save error: " << (err.empty() ? "none" : err);

    rllm::OutputLayer loaded;
    std::string load_err;
    loaded.load_from_safetensors(sf_file, &load_err);
    EXPECT_TRUE(load_err.empty()) << "Safetensors load error: " << (load_err.empty() ? "none" : load_err);

    // Compare via JSON serialization to avoid private member access.
    if (layer.save().dump() != loaded.save().dump()) ADD_FAILURE() << "json mismatch";

    std::filesystem::remove(sf_file);
}

// ---------------------------------------------------------------------------
// Safetensors model round-trip: checkpoint-style .st extension
// ---------------------------------------------------------------------------

TEST(SerializationTest, TextTrainerShortSafetensorsExtensionRoundTrip)
{
    rllm::Corpus corpus({});
    rllm::Statistics stats;
    rllm::TextTrainer network(0, corpus, stats);

    const std::string sf_file = (std::filesystem::temp_directory_path() / "checkpoint-short-extension.st").string();
    network.save(sf_file);
    ASSERT_TRUE(std::filesystem::exists(sf_file));

    rllm::TextTrainer loaded(0, corpus, stats);
    EXPECT_TRUE(loaded.load(sf_file));

    std::filesystem::remove(sf_file);
}

TEST(SerializationTest, CheckpointAliasUsesRelativeModelAndSidecarSymlinks)
{
    rllm::Corpus corpus({});
    rllm::Statistics stats;
    rllm::TextTrainer network(0, corpus, stats);
    const auto directory = std::filesystem::temp_directory_path() / "rllm-checkpoint-alias-test";
    std::filesystem::create_directories(directory);
    const auto checkpoint = directory / "checkpoint-123.st";
    const auto sidecar = std::filesystem::path(checkpoint.string() + ".training.json");
    const auto alias = directory / "checkpoint-best-window.st";
    const auto alias_sidecar = std::filesystem::path(alias.string() + ".training.json");
    {
        std::ofstream(checkpoint) << "model";
        std::ofstream(sidecar) << "parameters";
        std::ofstream(alias) << "old model";
        std::ofstream(alias_sidecar) << "old parameters";
    }

    network.link_checkpoint_alias(checkpoint, alias);

    EXPECT_TRUE(std::filesystem::is_symlink(alias));
    EXPECT_EQ(std::filesystem::read_symlink(alias), checkpoint.filename());
    EXPECT_TRUE(std::filesystem::is_symlink(alias_sidecar));
    EXPECT_EQ(std::filesystem::read_symlink(alias_sidecar), sidecar.filename());
    std::filesystem::remove_all(directory);
}

TEST(SerializationTest, TextTrainerJsonLoadCanExtendTransformerBlocks)
{
    rllm::Corpus corpus({});
    rllm::Statistics stats;
    rllm::TextTrainer source(0, corpus, stats);

    const std::string json_file = (std::filesystem::temp_directory_path() / "checkpoint-zero-blocks.json").string();
    source.save(json_file);
    ASSERT_TRUE(std::filesystem::exists(json_file));

    rllm::TextTrainer loaded(1, corpus, stats);
    EXPECT_TRUE(loaded.load(json_file));
    EXPECT_EQ(loaded.get_transformer_block_count(), 1u);

    std::filesystem::remove(json_file);
}

TEST(SerializationTest, TextTrainerSafetensorsStoresTransformerLoop)
{
    rllm::Corpus corpus({});
    rllm::Statistics stats;
    rllm::TextTrainer source(2, corpus, stats);

    const std::string sf_file = (std::filesystem::temp_directory_path() / "checkpoint-two-blocks.st").string();
    source.save(sf_file);
    ASSERT_TRUE(std::filesystem::exists(sf_file));

    safetensors::safetensors_t checkpoint;
    std::string warning;
    std::string error;
    ASSERT_TRUE(safetensors::load_from_file(sf_file, &checkpoint, &warning, &error)) << error;
    std::string first;
    std::string last;
    std::string count;
    ASSERT_TRUE(checkpoint.metadata.at("loop_first_block", &first));
    ASSERT_TRUE(checkpoint.metadata.at("loop_last_block", &last));
    ASSERT_TRUE(checkpoint.metadata.at("loop_count", &count));
    EXPECT_EQ(first, "0");
    EXPECT_EQ(last, "0");
    EXPECT_EQ(count, "1");

    rllm::TextTrainer loaded(2, corpus, stats);
    EXPECT_TRUE(loaded.load(sf_file));
    EXPECT_EQ(loaded.get_transformer_block_count(), 2u);
    EXPECT_EQ(loaded.get_loop_first_block(), 0u);
    EXPECT_EQ(loaded.get_loop_last_block(), 0u);
    EXPECT_EQ(loaded.get_loop_count(), 1u);
    EXPECT_EQ(
        loaded.get_transformer_execution_order(),
        (std::vector<size_t>{0, 1}));

    std::filesystem::remove(sf_file);
}

TEST(SerializationTest, SafetensorsTrainingStateDoesNotOverrideActiveLearningRateConfiguration)
{
    rllm::Corpus corpus({});
    rllm::Statistics stats;
    rllm::TextTrainer source(0, corpus, stats);
    const std::string sf_file = (std::filesystem::temp_directory_path() / "checkpoint-config-precedence.st").string();
    source.save(sf_file);

    safetensors::safetensors_t checkpoint;
    std::string warning;
    std::string error;
    ASSERT_TRUE(safetensors::load_from_file(sf_file, &checkpoint, &warning, &error)) << error;
    checkpoint.metadata.insert("training_state_version", "2");
    checkpoint.metadata.insert("training.optimizer", "muon+adamw");
    checkpoint.metadata.insert("training.optimizer_step", "7");
    checkpoint.metadata.insert("training.learning_rate", "0.01");
    checkpoint.metadata.insert("training.layer_learning_rate_multiplier", "1.1");
    checkpoint.metadata.insert("training.warmup_percent", "20");
    checkpoint.metadata.insert("training.learning_rate_schedule", "2");
    checkpoint.metadata.insert("training.learning_rate_schedule_steps", "100");
    checkpoint.metadata.insert("training.simulated_annealing_decay_factor", "0.7");
    checkpoint.metadata.insert("training.simulated_annealing_initial_multiplier", "8");
    checkpoint.metadata.insert("training.simulated_annealing_decay_epochs", "3");
    checkpoint.metadata.insert("training.simulated_annealing_min_multiplier", "0.03");
    checkpoint.metadata.insert("training.learning_rate_step", "7");
    checkpoint.metadata.insert("training.current_learning_rate", "0.08");
    checkpoint.metadata.insert("training.epochs_at_current_rate", "1");
    checkpoint.metadata.insert("training.epoch", "2");
    checkpoint.metadata.insert("training.window", "4");
    checkpoint.metadata.insert("training.window_count", "8");
    checkpoint.metadata.insert("training.rng_state", "");
    ASSERT_TRUE(safetensors::save_to_file(checkpoint, sf_file, &warning, &error)) << error;

    rllm::TextTrainer loaded(0, corpus, stats);
    loaded.set_learning_rate(0.02f);
    loaded.set_layer_learning_rate_multiplier(1.2f);
    loaded.set_warmup_percent(5.0f);
    loaded.set_learning_rate_schedule(rllm::LearningRateSchedule::Lowering);
    loaded.set_simulated_annealing_decay_factor(0.6f);
    loaded.set_simulated_annealing_initial_multiplier(4.0f);
    loaded.set_simulated_annealing_decay_epochs(2);
    loaded.set_simulated_annealing_min_multiplier(0.05f);

    ASSERT_TRUE(loaded.load(sf_file));
    EXPECT_FLOAT_EQ(loaded.get_learning_rate(), 0.02f);
    EXPECT_FLOAT_EQ(loaded.get_layer_learning_rate_multiplier(), 1.2f);
    EXPECT_FLOAT_EQ(loaded.get_warmup_percent(), 5.0f);
    EXPECT_EQ(loaded.get_learning_rate_schedule(), rllm::LearningRateSchedule::Lowering);
    EXPECT_FLOAT_EQ(loaded.get_simulated_annealing_decay_factor(), 0.6f);
    EXPECT_FLOAT_EQ(loaded.get_simulated_annealing_initial_multiplier(), 4.0f);
    EXPECT_EQ(loaded.get_simulated_annealing_decay_epochs(), 2u);
    EXPECT_FLOAT_EQ(loaded.get_simulated_annealing_min_multiplier(), 0.05f);

    std::filesystem::remove(sf_file);
}

// ---------------------------------------------------------------------------
// Safetensors layer round-trip: TransformerBlock (JSON-only test)
// The raw safetensors API requires direct member access which is private.
// The JSON save/load test covers TransformerBlock serialization.
// ---------------------------------------------------------------------------



// ---------------------------------------------------------------------------
// Serialization speed benchmark: JSON vs safetensors (InputLayer)
// ---------------------------------------------------------------------------

#include <chrono>
#include <iostream>

TEST(SerializationTest, BenchmarkJsonVsSafetensorsSpeed)
{
    constexpr int kIters = 50;

    // --- Warm-up run ---
    {
        rllm::InputLayer layer;
        layer.set_random_embeddings();
        auto j = layer.save(); (void)j.dump(2);

        rllm::InputLayer layer2;
        layer2.set_random_embeddings();
        const std::string sf_file =
            (std::filesystem::temp_directory_path() / "benchmark_input.safetensors").string();
        std::string warn, err;
        layer2.save_to_safetensors(sf_file, &warn, &err);
        rllm::InputLayer loaded;
        std::string load_err;
        loaded.load_from_safetensors(sf_file, &load_err);
        std::filesystem::remove(sf_file);
    }

    // --- JSON benchmark ---
    rllm::InputLayer json_layer;
    json_layer.set_random_embeddings();
    auto json_j = json_layer.save();
    std::string json_str = json_j.dump(2);

    auto t_json_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
        rllm::InputLayer loaded;
        loaded.load(nlohmann::json::parse(json_str));
        rllm::embedding_row_t dummy;
        loaded.get_embedding(rllm::TokenID{0}, dummy);
    }
    auto t_json_end = std::chrono::high_resolution_clock::now();

    // --- Safetensors benchmark ---
    rllm::InputLayer sf_layer;
    sf_layer.set_random_embeddings();
    const std::string sf_file =
        (std::filesystem::temp_directory_path() / "benchmark_input.safetensors").string();
    std::string warn, err;
    sf_layer.save_to_safetensors(sf_file, &warn, &err);

    auto t_sf_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
        rllm::InputLayer loaded;
        std::string load_err;
        loaded.load_from_safetensors(sf_file, &load_err);
    }
    auto t_sf_end = std::chrono::high_resolution_clock::now();

    std::filesystem::remove(sf_file);

    auto dur_json =
        std::chrono::duration_cast<std::chrono::microseconds>(t_json_end - t_json_start).count();
    auto dur_sf =
        std::chrono::duration_cast<std::chrono::microseconds>(t_sf_end - t_sf_start).count();

    std::printf("\n=== Serialization speed (%d iterations each) ===\n", kIters);
    std::printf("  JSON       : %ld us total  (%.2f us/op)\n", dur_json, dur_json / (double)kIters);
    std::printf("  safetensors: %ld us total  (%.2f us/op)\n", dur_sf, dur_sf / (double)kIters);

    if (dur_json < dur_sf) {
        std::printf("  => JSON is %.1fx faster\n\n", (double)dur_sf / dur_json);
    } else {
        std::printf("  => safetensors is %.1fx faster\n\n", (double)dur_json / dur_sf);
    }

    SUCCEED(); // always pass — benchmark results are informational only
}

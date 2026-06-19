// Test serialization of layers directly without needing Corpus or full NN setup.
#include <InputLayer.hpp>
#include <OutputLayer.hpp>
#include <TransformerBlock.hpp>
#include <Corpus.hpp>
#include <Statistics.hpp>
#include <NeuralNetwork.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

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

TEST(SerializationTest, NeuralNetworkShortSafetensorsExtensionRoundTrip)
{
    rllm::Corpus corpus({});
    rllm::Statistics stats;
    rllm::NeuralNetwork network(0, corpus, stats);

    const std::string sf_file = (std::filesystem::temp_directory_path() / "checkpoint-short-extension.st").string();
    network.save(sf_file);
    ASSERT_TRUE(std::filesystem::exists(sf_file));

    rllm::NeuralNetwork loaded(0, corpus, stats);
    EXPECT_TRUE(loaded.load(sf_file));

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

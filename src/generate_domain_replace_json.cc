#include <LayerPrimitives.hpp>
#include <TransformerBlock.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using SearchReplace = std::pair<std::string, std::string>;

void append_enum_pair(std::vector<SearchReplace>& pairs, const std::string& key, size_t value)
{
    pairs.emplace_back(key, std::to_string(value));
}

void append_float_pair(std::vector<SearchReplace>& pairs, const std::string& key, float value)
{
    pairs.emplace_back(key, std::to_string(value));
}

void append_layer_primitives_enum_pairs(std::vector<SearchReplace>& pairs)
{
    append_enum_pair(pairs, "EmbeddingDimension::START", static_cast<size_t>(rllm::EmbeddingDimension::START));
    append_enum_pair(pairs, "EmbeddingDimension::MAX", static_cast<size_t>(rllm::EmbeddingDimension::MAX));

    append_enum_pair(pairs, "PositionIndex::START", static_cast<size_t>(rllm::PositionIndex::START));
    append_enum_pair(pairs, "PositionIndex::MAX", static_cast<size_t>(rllm::PositionIndex::MAX));
    append_enum_pair(
        pairs,
        "PositionIndex::UNKNOWN_POSITION_INDEX",
        static_cast<size_t>(rllm::PositionIndex::UNKNOWN_POSITION_INDEX)
    );

    append_enum_pair(pairs, "HeadsIndex::START", static_cast<size_t>(rllm::HeadsIndex::START));
    append_enum_pair(pairs, "HeadsIndex::MAX", static_cast<size_t>(rllm::HeadsIndex::MAX));

    append_enum_pair(
        pairs,
        "MultiTokenPredictionIndex::START",
        static_cast<size_t>(rllm::MultiTokenPredictionIndex::START)
    );
    append_enum_pair(
        pairs,
        "MultiTokenPredictionIndex::ONE",
        static_cast<size_t>(rllm::MultiTokenPredictionIndex::ONE)
    );
    append_enum_pair(
        pairs,
        "MultiTokenPredictionIndex::TWO",
        static_cast<size_t>(rllm::MultiTokenPredictionIndex::TWO)
    );
    append_enum_pair(
        pairs,
        "MultiTokenPredictionIndex::THREE",
        static_cast<size_t>(rllm::MultiTokenPredictionIndex::THREE)
    );
    append_enum_pair(
        pairs,
        "MultiTokenPredictionIndex::FOUR",
        static_cast<size_t>(rllm::MultiTokenPredictionIndex::FOUR)
    );
    append_enum_pair(
        pairs,
        "MultiTokenPredictionIndex::MAX",
        static_cast<size_t>(rllm::MultiTokenPredictionIndex::MAX)
    );

    append_enum_pair(pairs, "HeadDimension::START", static_cast<size_t>(rllm::HeadDimension::START));
    append_enum_pair(pairs, "HeadDimension::MAX", static_cast<size_t>(rllm::HeadDimension::MAX));

    append_enum_pair(pairs, "FFDimension::START", static_cast<size_t>(rllm::FFDimension::START));
    append_enum_pair(pairs, "FFDimension::MAX", static_cast<size_t>(rllm::FFDimension::MAX));

    append_enum_pair(
        pairs,
        "NeuronConnectionIndex::START",
        static_cast<size_t>(rllm::NeuronConnectionIndex::START)
    );
    append_enum_pair(
        pairs,
        "NeuronConnectionIndex::MAX",
        static_cast<size_t>(rllm::NeuronConnectionIndex::MAX)
    );

    // Casts to enum types are not valid in generated shader code; force int casts.
    pairs.emplace_back("static_cast<EmbeddingDimension>", "(int)");
    pairs.emplace_back("static_cast<PositionIndex>", "(int)");
    pairs.emplace_back("static_cast<HeadsIndex>", "(int)");
    pairs.emplace_back("static_cast<MultiTokenPredictionIndex>", "(int)");
    pairs.emplace_back("static_cast<HeadDimension>", "(int)");
    pairs.emplace_back("static_cast<FFDimension>", "(int)");
    pairs.emplace_back("static_cast<NeuronConnectionIndex>", "(int)");
    pairs.emplace_back("static_cast<rllm::EmbeddingDimension>", "(int)");
    pairs.emplace_back("static_cast<rllm::PositionIndex>", "(int)");
    pairs.emplace_back("static_cast<rllm::HeadsIndex>", "(int)");
    pairs.emplace_back("static_cast<rllm::MultiTokenPredictionIndex>", "(int)");
    pairs.emplace_back("static_cast<rllm::HeadDimension>", "(int)");
    pairs.emplace_back("static_cast<rllm::FFDimension>", "(int)");
    pairs.emplace_back("static_cast<rllm::NeuronConnectionIndex>", "(int)");
}

void append_transformer_block_constant_pairs(std::vector<SearchReplace>& pairs)
{
    append_float_pair(pairs, "TransformerBlock::MOMENTUM_BETA", rllm::TransformerBlock::MOMENTUM_BETA);
    append_float_pair(pairs, "TransformerBlock::GRAD_CLIP", rllm::TransformerBlock::GRAD_CLIP);
    append_float_pair(pairs, "TransformerBlock::VEL_CLIP", rllm::TransformerBlock::VEL_CLIP);
    append_float_pair(pairs, "TransformerBlock::WEIGHT_CLAMP", rllm::TransformerBlock::WEIGHT_CLAMP);
    append_float_pair(pairs, "rllm::TransformerBlock::MOMENTUM_BETA", rllm::TransformerBlock::MOMENTUM_BETA);
    append_float_pair(pairs, "rllm::TransformerBlock::GRAD_CLIP", rllm::TransformerBlock::GRAD_CLIP);
    append_float_pair(pairs, "rllm::TransformerBlock::VEL_CLIP", rllm::TransformerBlock::VEL_CLIP);
    append_float_pair(pairs, "rllm::TransformerBlock::WEIGHT_CLAMP", rllm::TransformerBlock::WEIGHT_CLAMP);
}

void print_json_pairs(const std::vector<SearchReplace>& pairs)
{
    nlohmann::json json_pairs = nlohmann::json::array();
    for (const auto& [search, replace] : pairs)
    {
        json_pairs.push_back({{"search", search}, {"replace", replace}});
    }
    std::cout << json_pairs.dump(2) << '\n';
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: generate_domain_replace_json --print-json\n";
        return 1;
    }

    const std::string_view arg{argv[1]};
    if (arg == "--print-json")
    {
        std::vector<SearchReplace> pairs;
        append_layer_primitives_enum_pairs(pairs);
        append_transformer_block_constant_pairs(pairs);
        print_json_pairs(pairs);
        return 0;
    }

    std::cerr << "unsupported query: " << arg << '\n';
    return 2;
}

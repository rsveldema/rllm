#pragma once

#include <LayerPrimitives.hpp>

#include <nlohmann/json_fwd.hpp>
#include <Corpus.hpp>

namespace rllm
{
    class OutputLayer
    {
      public:
        OutputLayer()
        {}
        ~OutputLayer() = default;
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer& operator=(const OutputLayer&) = delete;

        void set_random_weights_and_connections_for_output_layer(Corpus& corpus);

        void update_output_weights(const template_token_vector<float, TokenID>& delta, float learning_rate);

        void compute_score(Score& score, const TokenID expected_output_token);

        void compute_deltas(const Score& score, template_token_vector<float, TokenID>& deltas) const;
        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      public:
        template_token_vector<float, TokenID> m_inputs;
        template_token_vector<float, TokenID> m_trigger_values;
        template_token_vector<float, TokenID> m_weights;
    };

} // namespace rllm

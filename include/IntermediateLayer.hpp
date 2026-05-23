#pragma once

#include <LayerPrimitives.hpp>
#include <Corpus.hpp>
#include <OutputLayer.hpp>

#include <algorithm>
#include <nlohmann/json_fwd.hpp>


namespace rllm
{
    class IntermediateLayer
    {
      public:
        IntermediateLayer(Corpus& corpus)
            : m_corpus(corpus)
        {}
        ~IntermediateLayer() = default;

        void propagate_forward(IntermediateLayer& next_layer);
        void propagate_forward_to_output(OutputLayer& output_layer) const;

        void fill_inputs(float value) { m_inputs.fill(value); }
        bool count_fires(IntermediateLayerIndex i) const { return m_inputs[i] >= m_trigger_values[i]; }
        void accumulate_input(IntermediateLayerIndex index, float value, Range<float> range)
        {
            m_inputs.add_with_clamp(index, value, range);
        }

        void propagate_backward(
            const template_token_vector<float, IntermediateLayerIndex>& delta,
            template_token_vector<float, IntermediateLayerIndex>& prev_delta,
            float learning_rate
        );

        void propagate_backward_from_output_layer(
            const template_token_vector<float, TokenID>& delta,
            template_token_vector<float, IntermediateLayerIndex>& prev_delta,
            float learning_rate
        );

        void set_random_weights_and_connections();
        void set_random_weights_and_connections_to_output_layer();
        void load(const nlohmann::json& j);
        nlohmann::json save() const;



        void clear_last_weight_deltas()
        {
            m_last_weight_delta.fill(0.0f);
        }

      private:
        Corpus& m_corpus;
        // accumulated input for each neuron in the layer
        template_token_vector<float, IntermediateLayerIndex> m_inputs;
        // if m_inputs[i] >= m_trigger_values[i], then neuron 'i' fires.
        // this value is learned during training and can be thought of as the "bias"
        // for the neuron.
        template_token_vector<float, IntermediateLayerIndex> m_trigger_values;
        // the weight of the connection from neuron 'i' in this layer
        // to neuron 'm_connections[i]' in the next layer
        template_token_vector<float, IntermediateLayerIndex> m_weights;
        // weight delta applied during the last backprop step (positive = weight increased)
        template_token_vector<float, IntermediateLayerIndex> m_last_weight_delta;
        // neuron 'n' is connected to one or more neurons in the next layer
        template_token_vector<std::vector<IntermediateLayerIndex>, IntermediateLayerIndex> m_connections;


        // private methods made public for testing purposes
        void randomize_neuron(IntermediateLayerIndex i);
        void randomize_neuron_to_output(IntermediateLayerIndex i);
        void forward_neuron(IntermediateLayerIndex i, IntermediateLayer& next_layer);
        void forward_neuron_to_output(IntermediateLayerIndex i, OutputLayer& output_layer) const;
        void backward_neuron(
            IntermediateLayerIndex i,
            const template_token_vector<float, IntermediateLayerIndex>& delta,
            template_token_vector<float, IntermediateLayerIndex>& prev_delta,
            float learning_rate
        );
        void backward_neuron_from_output(
            IntermediateLayerIndex i,
            const template_token_vector<float, TokenID>& delta,
            template_token_vector<float, IntermediateLayerIndex>& prev_delta,
            float learning_rate
        );
    };

} // namespace rllm

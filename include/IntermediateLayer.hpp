#pragma once

#include <LayerPrimitives.hpp>
#include <Corpus.hpp>
#include <OutputLayer.hpp>

#include <algorithm>
#include <cmath>
#include <nlohmann/json_fwd.hpp>


namespace rllm
{
    struct OutConnection {
        // target neuron can be either a neuron in the next intermediate layer or a neuron in the output layer.
        // We can distinguish between the two cases based on which layer it is in. The last intermediate layer
        // will only have connections to the output layer, and all other intermediate layers will only have connections
        // to the next intermediate layer.
        IntermediateLayerIndex target_neuron;

        // the weight of the connection from the current neuron to the target neuron in the next layer.
        // this is learned during training and can be thought of as the "weight" for the connection.
        float weight;
    };

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

      private:
        Corpus& m_corpus;
        // neuron 'i's accumulated input for each neuron in the layer
        template_token_vector<float, IntermediateLayerIndex> m_inputs;
        // neuron 'i' is connected to one or more neurons in the next layer
        template_token_vector<std::vector<OutConnection>,
            IntermediateLayerIndex> m_connections;


        void randomize_neuron(IntermediateLayerIndex i);

        void randomize_neuron_to_output(IntermediateLayerIndex i);

        TokenID  get_random_target_neuron_for_output_layer(IntermediateLayerIndex from_neuron);

        IntermediateLayerIndex get_random_target_neuron_for_intermediate_layer(IntermediateLayerIndex from_neuron);



        void forward_neuron(IntermediateLayerIndex i, IntermediateLayer& next_layer) const;
        void forward_neuron_to_output(IntermediateLayerIndex i, OutputLayer& output_layer) const;



        bool have_connection_to_neuron(IntermediateLayerIndex from_neuron, IntermediateLayerIndex to_neuron) const
        {
            for (const auto& connection : m_connections[from_neuron])
            {
                if (connection.target_neuron == to_neuron)
                    return true;
            }
            return false;
        }
        bool have_connection_to_neuron(IntermediateLayerIndex from_neuron, TokenID to_neuron) const
        {
            for (const auto& connection : m_connections[from_neuron])
            {
                if (static_cast<TokenID>(connection.target_neuron) == to_neuron)
                    return true;
            }
            return false;
        }

        float normal_activation_function(float x) const
        {
            // simple ReLU activation function. You can experiment with other activation functions if you like.
            return std::max(0.0f, x);
        }

        float outputlayer_activation_function(float x) const
        {
            // Gaussian activation centred at 0 with sigma=1: peaks at 1 when x==0
            // and decays symmetrically, giving a smooth bounded output in (0,1].
            return std::exp(-(x * x) / 2.0f);
        }
    };

} // namespace rllm

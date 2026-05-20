#pragma once

#include <Corpus.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace rllm
{
    // position of a token in the input sequence. For example, in the input "the cat sat", the token "cat" has
    // position 1.
    enum class PositionIndex : size_t
    {
        START = 0,
        MAX = 32,
        UNKNOWN_POSITION_INDEX = static_cast<size_t>(-1)
    };

    // position of a neuron in the intermediate layer. For example, in the intermediate layer, neuron 0 is connected to
    // token 0 in the input layer, neuron 1 is connected to token 1 in the input layer, and so on.
    enum class IntermediateLayerIndex : size_t
    {
        START = 0,
        MAX = 1024 * 8,
        UNKNOWN_INTERMEDIATE_LAYER_INDEX = static_cast<size_t>(-1)
    };


    static inline PositionIndex inc(PositionIndex id)
    {
        assert(id != PositionIndex::UNKNOWN_POSITION_INDEX);
        assert(id < PositionIndex::MAX);
        return static_cast<PositionIndex>(static_cast<int32_t>(id) + 1);
    }

    static inline IntermediateLayerIndex inc(IntermediateLayerIndex id)
    {
        assert(id != IntermediateLayerIndex::UNKNOWN_INTERMEDIATE_LAYER_INDEX);
        assert(id < IntermediateLayerIndex::MAX);
        return static_cast<IntermediateLayerIndex>(static_cast<int32_t>(id) + 1);
    }


    template <typename T, typename LengthType>
    class template_token_vector
    {
      public:
        template_token_vector()
        {
            m_data.fill(T{});
        }
        ~template_token_vector() = default;
        template_token_vector(const template_token_vector&) = delete;
        template_token_vector& operator=(const template_token_vector&) = delete;

        T& operator[](LengthType index)
        {
            return m_data[static_cast<size_t>(index)];
        }

        const T& operator[](LengthType index) const
        {
            return m_data[static_cast<size_t>(index)];
        }

        void fill(T value)
        {
            m_data.fill(value);
        }

      private:
        using token_vector_data_t = std::array<T, static_cast<size_t>(LengthType::MAX)>;
        token_vector_data_t m_data;
    };

    template <typename T, typename X, typename Y>
    class template_token_matrix
    {
      public:
        template_token_matrix()
        {
            for (auto& row : m_data)
            {
                row.fill(T{});
            }
        }
        ~template_token_matrix() = default;
        template_token_matrix(const template_token_matrix&) = delete;

        void set(const X x, const Y y, T value)
        {
            auto& inner_data = m_data[static_cast<size_t>(x)];
            inner_data[static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, T value)
        {
            set(indices.first, indices.second, value);
        }

        const T& get(const X x, const Y y) const
        {
            return m_data[static_cast<size_t>(x)][static_cast<size_t>(y)];
        }

        const T& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        void fill(T value)
        {
            for (auto& row : m_data)
            {
                row.fill(value);
            }
        }

      private:
        using inner_array_t = std::array<T, static_cast<size_t>(Y::MAX)>;
        using matrix_data_t = std::array<inner_array_t, static_cast<size_t>(X::MAX)>;
        matrix_data_t m_data;
    };


    struct Score
    {
        template_token_vector<float, TokenID> values;
    };

    struct OutputToken
    {
        TokenID token_id;
        float activation;
        float weight;
    };

    class InputLayer
    {
      public:
        InputLayer()
        {}
        ~InputLayer() = default;
        InputLayer(const InputLayer&) = delete;
        InputLayer& operator=(const InputLayer&) = delete;

        void set_input_layer(const InputLine& input);

        void set_input(const TokenID token_id, const PositionIndex pos_index, float value)
        {
            m_inputs.set(token_id, pos_index, value);
        }

        void set_random_weights_and_connections();

      private:
        template_token_matrix<float, TokenID, PositionIndex> m_inputs;
        template_token_matrix<float, TokenID, PositionIndex> m_trigger_values;
        template_token_matrix<float, TokenID, PositionIndex> m_weights;

        // neuron 'i' is connected to neuron 'm_connections[i,j]' in the next layer
        template_token_matrix<std::pair<IntermediateLayerIndex, PositionIndex>, TokenID, PositionIndex>
            m_connections;
    };

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

       void compute_deltas(const Score& score, template_token_vector<float, TokenID> &deltas) const;


      public:
        template_token_vector<float, TokenID> m_inputs;
        template_token_vector<float, TokenID> m_trigger_values;
        template_token_vector<float, TokenID> m_weights;
    };

    class IntermediateLayer
    {
      public:
        IntermediateLayer()
        {}
        ~IntermediateLayer() = default;
        IntermediateLayer(const IntermediateLayer&) = delete;
        IntermediateLayer& operator=(const IntermediateLayer&) = delete;
        IntermediateLayer(IntermediateLayer&&) = default;
        IntermediateLayer& operator=(IntermediateLayer&&) = delete;

        void propagate_forward(IntermediateLayer& next_layer);
        void propagate_backward(
            const template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& delta,
            template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& prev_delta,
            float learning_rate
        );

        void propagate_backward(
            const  template_token_vector<float, TokenID>& delta,
            template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& prev_delta,
            float learning_rate
        );

        void set_random_weights_and_connections();
        void set_random_weights_and_connections_to_output_layer(Corpus& corpus);

      private:
        // accumulated input for each neuron in the layer
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex> m_inputs;
        // if m_inputs[i] >= m_trigger_values[i], then neuron 'i' fires.
        // this value is learned during training and can be thought of as the "bias"
        // for the neuron.
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex> m_trigger_values;
        // the weight of the connection from neuron 'i' in this layer
        // to neuron 'm_connections[i]' in the next layer
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex> m_weights;
        // neuron 'n' is connected to neuron 'm_connections[i,j]' in the next layer
        template_token_matrix<std::pair<IntermediateLayerIndex, PositionIndex>, IntermediateLayerIndex, PositionIndex>
            m_connections;

        friend class NeuralNetwork;
    };


    class NeuralNetwork
    {
      public:
        NeuralNetwork(size_t num_layers)
            : m_intermediate_layers(num_layers)
        {}
        ~NeuralNetwork() = default;
        NeuralNetwork(const NeuralNetwork&) = delete;
        NeuralNetwork& operator=(const NeuralNetwork&) = delete;

        void compute_score(Score& score, const TokenID expected_output_token);
        void propagate_backward(const Score& score);
        void propagate_forward();

        void set_input_layer(const InputLine& input);

        // returns the top-K with the biggest activation in the output layer, as pairs of (token_id, activation_value)
        std::vector<OutputToken> get_best_output_token_ids(size_t top_k, Corpus& corpus) const;


        void train(Corpus& corpus);
        void set_random_weights_and_connections(Corpus& corpus);

        void load(const std::string& filename);
        void save(const std::string& filename) const;

      private:
        InputLayer m_input_layer;
        std::vector<IntermediateLayer> m_intermediate_layers;
        OutputLayer m_output_layer;
    };

    class RLLM
    {
      public:
        RLLM();
        ~RLLM() = default;
        RLLM(const RLLM&) = delete;
        RLLM& operator=(const RLLM&) = delete;

        void train_mode(const std::string& filename);
        void prompt_mode(const std::string& filename);
    };

} // namespace rllm
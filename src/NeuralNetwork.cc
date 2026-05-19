#include <RLLM.hpp>
#include <algorithm>
#include <cassert>
#include <print>

namespace rllm {

// Helper functions

static size_t get_random_value_centered_around(size_t center,
                                               size_t range = 10) {
  int k = rand() % (2 * range + 1) - range;
  if ((static_cast<int>(center) + k) < 0)
    return 0;
  return static_cast<size_t>(static_cast<int>(center) + k);
}

static size_t clip_max(size_t value, size_t max) {
  if (value > max)
    return max;
  return value;
}

static float get_random_value() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

static TokenID neuron_to_token_id(size_t neuron_index) {
  return static_cast<TokenID>(neuron_index % TOKENS_PER_POSITION);
}

static size_t token_id_to_neuron(TokenID token_id, size_t position) {
  return position * TOKENS_PER_POSITION + (token_id % TOKENS_PER_POSITION);
}

// Layer

void Layer::set_input_layer(const InputLine &input) {
  m_inputs.fill(0.0f);
  for (size_t i = 0; i < input.size() && i < MAX_INPUT_POSITIONS; ++i) {
    m_inputs[token_id_to_neuron(input[i], i)] = 1.0f;
  }
}

void Layer::set_random_weights_and_connections() {
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    m_trigger_values[i] = get_random_value();
    m_weights[i] = get_random_value();
    m_connections[i] =
        clip_max(get_random_value_centered_around(i), LAYER_SIZE - 1);
  }
}


void Layer::set_random_weights_and_connections_to_output_layer(Corpus& corpus) {
  // setup the layer JUST before the output layer.
  // It needs to have connections to the output layer that are distributed across the tokens in the corpus.
  for (size_t i = 0; i < LAYER_SIZE; ++i) {
    m_trigger_values[i] = get_random_value();
    m_weights[i] = get_random_value();
    m_connections[i] = i % corpus.size();
  }
}

void Layer::set_random_weights_and_connections_for_output_layer(Corpus& corpus) {
  // setup the output layer itself. It has no connections to other neurons.
  for (size_t i = 0; i < LAYER_SIZE; ++i) {
    m_trigger_values[i] = get_random_value();
    m_weights[i] = get_random_value();
    m_connections[i] = -1;
  }
}

void Layer::propagate_forward(Layer &next_layer) {
  next_layer.m_inputs.fill(0.0f);
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    if (m_inputs[i] >= m_trigger_values[i]) {
      const auto next_neuron_index = m_connections[i];
      assert(next_neuron_index != static_cast<size_t>(-1));

      const auto weight = m_weights[i];
      next_layer.m_inputs[next_neuron_index] = std::clamp(
          next_layer.m_inputs[next_neuron_index] + weight * m_inputs[i], 0.0f,
          1.0f);
    }
  }
}

void Layer::update_output_weights(const vector<float> &delta,
                                  float learning_rate) {
  for (size_t i = 0; i < LAYER_SIZE; ++i) {
    m_weights[i] = std::clamp(
        m_weights[i] + learning_rate * delta[i] * m_inputs[i], 0.0f, 1.0f);
    // Adjust trigger: lower when delta > 0 (fire more), raise when delta < 0.
    m_trigger_values[i] = std::clamp(
        m_trigger_values[i] - learning_rate * delta[i], 0.0f, 1.0f);
  }
}

void Layer::propagate_backward(const vector<float> &delta,
                               vector<float> &prev_delta, float learning_rate) {
  for (size_t i = 0; i < LAYER_SIZE; ++i) {
    if (m_inputs[i] < m_trigger_values[i])
      continue; // neuron did not fire, no gradient to propagate

    const size_t k = m_connections[i];
    assert(k != static_cast<size_t>(-1));
    const float d = delta[k];

    // Increase weight when downstream error is positive (need more signal).
    m_weights[i] =
        std::clamp(m_weights[i] + learning_rate * d * m_inputs[i], 0.0f, 1.0f);

    // Lower trigger makes this neuron fire more easily — helpful when
    // downstream error is positive.
    m_trigger_values[i] =
        std::clamp(m_trigger_values[i] - learning_rate * d, 0.0f, 1.0f);

    // Accumulate gradient for the layer below.
    prev_delta[i] += d * m_weights[i];
  }
}

// NeuralNetwork

static void try_add_to_top_k(std::vector<OutputToken> &top_k, TokenID id,
                             float value, float weight, size_t k) {
  if (top_k.size() < k) {
    top_k.push_back({id, value, weight});
    std::sort(top_k.begin(), top_k.end(), [](const auto &a, const auto &b) {
      return a.activation > b.activation;
    });
  } else if (!top_k.empty() && value > top_k.back().activation) {
    top_k.back() = {id, value, weight};
    std::sort(top_k.begin(), top_k.end(), [](const auto &a, const auto &b) {
      return a.activation > b.activation;
    });
  }
}

void NeuralNetwork::set_input_layer(const InputLine &input) {
  assert(!m_layers.empty());
  m_layers[0].set_input_layer(input);
}

// returns the top-K with the biggest activation in the output layer,
// sorted descending by activation
std::vector<OutputToken>
NeuralNetwork::get_best_output_token_ids(size_t top_k, Corpus& corpus) const {
  assert(!m_layers.empty());

  std::vector<OutputToken> top_k_pairs;
  const auto &output_layer = m_layers.back();
  for (size_t i = 1; i < corpus.size(); ++i) {
    if (output_layer.m_inputs[i] < output_layer.m_trigger_values[i])
      continue; // neuron did not fire
    try_add_to_top_k(top_k_pairs,
                     static_cast<TokenID>(i),
                     output_layer.m_inputs[i], output_layer.m_weights[i],
                     top_k);
  }
  return top_k_pairs;
}

void NeuralNetwork::compute_score(Score &score,
                                  const TokenID expected_output_token) {
  // Store the target for each output neuron:
  //   1.0 = this neuron should fire (expected token)
  //   0.0 = this neuron should NOT fire (wrong token)
  for (size_t i = 0; i < LAYER_SIZE; ++i) {
    score.values[i] = (i == expected_output_token) ? 1.0f : 0.0f;
  }
}

void NeuralNetwork::propagate_backward(const Score &score) {
  if (m_layers.size() < 2)
    return;

  static constexpr float LEARNING_RATE = 0.01f;

  // delta[i] = signed error: target - actual.
  // Positive  → neuron fires too little  → increase weight.
  // Negative  → neuron fires too much    → decrease weight.
  const auto &out = m_layers.back();
  vector<float> delta;
  for (size_t i = 0; i < LAYER_SIZE; ++i)
    delta[i] = score.values[i] - out.m_inputs[i];

  // Update the output layer's weights directly from the score delta.
  m_layers.back().update_output_weights(delta, LEARNING_RATE);

  // Walk backwards through the remaining layers.
  for (int l = static_cast<int>(m_layers.size()) - 2; l >= 0; --l) {
    Layer &layer = m_layers[l];
    vector<float> prev_delta;
    prev_delta.fill(0.0f);

    layer.propagate_backward(delta, prev_delta, LEARNING_RATE);

    delta = prev_delta;
  }
}

void NeuralNetwork::set_random_weights_and_connections(Corpus& corpus) {
  for (size_t i = 0; i < (m_layers.size()-1); ++i) {
    m_layers[i].set_random_weights_and_connections();
  }
  m_layers[m_layers.size()-1].set_random_weights_and_connections_to_output_layer(corpus);
  m_layers.back().set_random_weights_and_connections_for_output_layer(corpus);
}

void NeuralNetwork::train(Corpus &corpus) {
  std::println("Training the neural network...");

  set_random_weights_and_connections(corpus);

  // Get a training example from the corpus. The example needs at least 2
  // tokens. Note that the list can be padded to 2 tokens using
  // UNKNOWN_TOKEN_ID if necessary.
  const auto train_output = corpus.get_training_input_line(2);
  auto train_input = train_output;
  const auto expected_output_token = train_input.back();
  train_input.pop_back();

  const auto full_string = corpus.get_line(train_output);

  size_t num_iterations = 1000000;
  for (size_t i = 0; i < num_iterations; ++i) {

    Score score;
    set_input_layer(train_input);
    propagate_forward();
    compute_score(score, expected_output_token);
    propagate_backward(score);

    if (i % 100 == 0) {
        const auto expected_token =
            corpus.get_token_from_id(expected_output_token);
      std::println("Training iteration[{}], wanted: '{}' ({}), full string: '{}'", i,
                expected_token,
                   expected_output_token, full_string);

      int prediction_index = 0;
      const auto predicted_token_id_lists = get_best_output_token_ids(5, corpus);
      for (const auto &entry : predicted_token_id_lists) {
        const auto predicted_token = corpus.get_token_from_id(entry.token_id);
        std::println("\t predicted: {} / pred:'{}' (id: '{}'), {} (weight: {})",
                     prediction_index, predicted_token, entry.token_id,
                     entry.activation, entry.weight);
        prediction_index++;
      }
    }
  }
}

} // namespace rllm

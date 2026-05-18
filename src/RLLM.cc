#include <RLLM.hpp>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <print>
#include <string>

const size_t num_layers = 3; // Example number of layers

namespace rllm {
RLLM::RLLM() {
  // Constructor implementation
}

void RLLM::prompt_mode(const std::string &filename) {
  Corpus corpus;

  NeuralNetwork nn(num_layers);
  nn.load(filename);

  std::string line;
  while (true) {
    std::cout << "Enter input (or 'exit' to quit): ";
    if (!std::getline(std::cin, line) || line == "exit") {
      break;
    }
    // Process the input line
    auto token_id_list = corpus.get_token_ids(line);

    for (int i = 0; i < token_id_list.size(); ++i) {
      // Set the input layer of the neural network based on the token IDs
      nn.set_input_layer(token_id_list[i]);

      // Propagate through the network to get the output
      nn.propagate_forward();

      // Get the output and convert it back to a token
      const auto output_token_id = nn.get_output_token_id();
      const auto output_token = corpus.get_token_from_id(output_token_id);

      std::println("Predicted next token: {}", output_token);

      token_id_list.push_back(
          output_token_id); // Add the predicted token ID to the input for the
                            // next iteration
    }
  }
}

size_t get_random_value_centered_around(size_t center, size_t range = 10) {
  int k = rand() % (2 * range + 1) -
          range; // Random value between -range and +range
  unsigned result = static_cast<int>(center) + k;
  return result;
}

size_t clip_max(size_t value, size_t max) {
  if (value > max)
    return max;
  return value;
}

/** returns a random float between 0 and 1 */
float get_random_value() {
  float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  return random;
}

  void NeuralNetwork::load(const std::string &filename)
  {
    std::ifstream file{filename};
    if (!file) return;
    const auto j = nlohmann::json::parse(file);

    m_layers.clear();
    for (const auto& jlayer : j["layers"]) {
      Layer layer;
      const auto inputs      = jlayer["inputs"].get<std::vector<float>>();
      const auto triggers    = jlayer["trigger_values"].get<std::vector<float>>();
      const auto weights     = jlayer["weights"].get<std::vector<float>>();
      const auto connections = jlayer["connections"].get<std::vector<size_t>>();
      std::copy(inputs.begin(),      inputs.end(),      layer.m_inputs.begin());
      std::copy(triggers.begin(),    triggers.end(),    layer.m_trigger_values.begin());
      std::copy(weights.begin(),     weights.end(),     layer.m_weights.begin());
      std::copy(connections.begin(), connections.end(), layer.m_connections.begin());
      m_layers.push_back(std::move(layer));
    }
  }

  void NeuralNetwork::save(const std::string &filename) const
  {
    nlohmann::json j;
    j["layers"] = nlohmann::json::array();
    for (const auto& layer : m_layers) {
      nlohmann::json jlayer;
      jlayer["inputs"]        = std::vector<float>(layer.m_inputs.begin(),       layer.m_inputs.end());
      jlayer["trigger_values"]= std::vector<float>(layer.m_trigger_values.begin(),layer.m_trigger_values.end());
      jlayer["weights"]       = std::vector<float>(layer.m_weights.begin(),      layer.m_weights.end());
      jlayer["connections"]   = std::vector<size_t>(layer.m_connections.begin(), layer.m_connections.end());
      j["layers"].push_back(std::move(jlayer));
    }
    std::ofstream file{filename};
    file << j.dump(2) << '\n';
  }


void NeuralNetwork::compute_score(Score &score) { score.values.fill(0.0f); }

void NeuralNetwork::propagate_backward(const Score &score) {
  // Placeholder implementation
}

void Layer::set_random_weights_and_connections() {
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    m_trigger_values[i] =
        get_random_value(); // Random trigger value between 0 and 1
    m_weights[i] = get_random_value();

    // we use get_random_value_centered_around to create some locality in the
    // connections between layers
    m_connections[i] = clip_max(
        get_random_value_centered_around(i),
        LAYER_SIZE - 1); // Random connection to a neuron in the next layer
  }
}

void NeuralNetwork::set_random_weights_and_connections() {
  for (size_t i = 0; i < m_layers.size() - 1; ++i) {
    m_layers[i]
        .set_random_weights_and_connections(); // Set random weights and
                                               // connections for layer 'i'
  }
}

void Layer::propagate_forward(Layer &next_layer) {
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    if (m_inputs[i] >= m_trigger_values[i]) {
      // neuron 'i' fires, so we add its weight to the input of the connected
      // neuron in the next layer
      const auto next_neuron_index = m_connections[i];
      const auto weight = m_weights[i];

      next_layer.m_inputs[next_neuron_index] = weight * m_inputs[i];
    }
  }
}

void NeuralNetwork::train() {
  std::println("Training the neural network...");

  set_random_weights_and_connections();

  size_t num_iterations = 100; // Example number of training iterations
  for (size_t i = 0; i < num_iterations; ++i) {
    // Simulate training by printing the iteration number
    std::println("Training iteration: {}", i);
    propagate_forward();

    Score score;
    compute_score(score);
    propagate_backward(score);
  }
}

void RLLM::train_mode(const std::string &filename) {
  std::println("Training mode");

  Corpus corpus;

  NeuralNetwork nn(num_layers);

  nn.train();

  nn.save(filename);
}
} // namespace rllm
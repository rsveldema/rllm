#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace rllm {

using TokenID = uint32_t;
using Token = std::string;

class Corpus {
public:
  Corpus();

private:
  std::map<Token, TokenID> m_token_to_id;

  class TokenData {
  public:
    explicit TokenData(std::string filename) : filename(std::move(filename)) {}

    void add(TokenID id) {
        data.push_back(id);
    }

  private:
    std::string filename;
    std::vector<TokenID> data; // positions of the token in the corpus
  };

  std::vector<TokenData> m_token_list;
};

static constexpr size_t LAYER_SIZE = 128;
template <typename T>

using vector = std::array<T, LAYER_SIZE>;

struct Score {
  vector<float> values;
};

class Layer {
public:
  Layer() {}
  ~Layer() = default;
  Layer(const Layer &) = delete;
  Layer &operator=(const Layer &) = delete;

  void propagate_forward(Layer &next_layer);

  void set_random_weights_and_connections();

private:
  // accumulated input for each neuron in the layer
  vector<float> m_inputs;
  // if m_inputs[i] >= m_trigger_values[i], then neuron 'i' fires.
  // this value is learned during training and can be thought of as the "bias"
  // for the neuron.
  vector<float> m_trigger_values;
  // the weight of the connection from neuron 'i' in this layer
  // to neuron 'm_connections[i]' in the next layer
  vector<float> m_weights;
  // neuron 'i' is connected to neuron 'm_connections[i]' in the next layer
  vector<size_t> m_connections;

  friend class NeuralNetwork;
};

class NeuralNetwork {
public:
  NeuralNetwork(size_t num_layers) : m_layers(num_layers) {}
  ~NeuralNetwork() = default;
  NeuralNetwork(const NeuralNetwork &) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &) = delete;

  void compute_score(Score &score);
  void propagate_backward(const Score &score);

  void propagate_forward() {
    for (size_t i = 0; i < m_layers.size() - 1; ++i) {
      m_layers[i].propagate_forward(m_layers[i + 1]);
    }
  }

  void train();
  void set_random_weights_and_connections();

  void load(const std::string &filename);
  void save(const std::string &filename) const;

private:
  std::vector<Layer> m_layers;
};

class RLLM {
public:
  RLLM();
  ~RLLM() = default;
  RLLM(const RLLM &) = delete;
  RLLM &operator=(const RLLM &) = delete;

  void train_mode(const std::string &filename);
  void prompt_mode(const std::string &filename);
};

} // namespace rllm
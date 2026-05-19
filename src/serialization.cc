#include <RLLM.hpp>

#include <fstream>
#include <nlohmann/json.hpp>

namespace rllm {

void NeuralNetwork::load(const std::string &filename) {
  std::ifstream file{filename};
  if (!file)
    return;
  const auto j = nlohmann::json::parse(file);

  m_layers.clear();
  for (const auto &jlayer : j["layers"]) {
    Layer layer;
    const auto inputs = jlayer["inputs"].get<std::vector<float>>();
    const auto triggers = jlayer["trigger_values"].get<std::vector<float>>();
    const auto weights = jlayer["weights"].get<std::vector<float>>();
    const auto connections = jlayer["connections"].get<std::vector<size_t>>();
    std::copy(inputs.begin(), inputs.end(), layer.m_inputs.begin());
    std::copy(triggers.begin(), triggers.end(), layer.m_trigger_values.begin());
    std::copy(weights.begin(), weights.end(), layer.m_weights.begin());
    std::copy(connections.begin(), connections.end(),
              layer.m_connections.begin());
    m_layers.push_back(std::move(layer));
  }
}

void NeuralNetwork::save(const std::string &filename) const {
  nlohmann::json j;
  j["layers"] = nlohmann::json::array();
  for (const auto &layer : m_layers) {
    nlohmann::json jlayer;
    jlayer["inputs"] =
        std::vector<float>(layer.m_inputs.begin(), layer.m_inputs.end());
    jlayer["trigger_values"] = std::vector<float>(
        layer.m_trigger_values.begin(), layer.m_trigger_values.end());
    jlayer["weights"] =
        std::vector<float>(layer.m_weights.begin(), layer.m_weights.end());
    jlayer["connections"] = std::vector<size_t>(layer.m_connections.begin(),
                                                layer.m_connections.end());
    j["layers"].push_back(std::move(jlayer));
  }
  std::ofstream file{filename};
  file << j.dump(2) << '\n';
}

void Corpus::save_token_map(const std::string &filename) const {
  nlohmann::json j;
  for (const auto &[token, id] : m_token_to_id) {
    j[token] = id;
  }
  std::ofstream file{filename};
  file << j.dump(2) << '\n';
}


} // namespace rllm

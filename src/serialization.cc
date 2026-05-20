#include <RLLM.hpp>

#include <fstream>
#include <nlohmann/json.hpp>

namespace rllm
{

    void NeuralNetwork::load(const std::string& filename)
    {
        std::ifstream file{filename};
        if (!file)
            return;
        const auto j = nlohmann::json::parse(file);

        m_intermediate_layers.clear();
    }

    void NeuralNetwork::save(const std::string& filename) const
    {
        nlohmann::json j;
        std::ofstream file{filename};
        file << j.dump(2) << '\n';
    }

    void Corpus::save_token_map(const std::string& filename) const
    {
        nlohmann::json j;
        for (const auto& [token, id] : m_token_to_id)
        {
            j[token] = id;
        }
        std::ofstream file{filename};
        file << j.dump(2) << '\n';
    }


} // namespace rllm

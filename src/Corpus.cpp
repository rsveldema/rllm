#include <RLLM.hpp>

#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string_view>

namespace rllm {

    static std::vector<std::string> tokenize(std::string_view text) {
        std::vector<std::string> tokens;
        std::string word;
        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!word.empty()) {
                    tokens.push_back(std::move(word));
                    word.clear();
                }
            } else if (std::ispunct(static_cast<unsigned char>(c))) {
                if (!word.empty()) {
                    tokens.push_back(std::move(word));
                    word.clear();
                }
                tokens.push_back(std::string(1, c));
            } else {
                word += c;
            }
        }
        if (!word.empty()) {
            tokens.push_back(std::move(word));
        }
        return tokens;
    }

    Corpus::Corpus() {
        const std::filesystem::path corpus_dir{"corpus"};
        if (!std::filesystem::exists(corpus_dir)) {
            return;
        }

        TokenID next_id = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(corpus_dir)) {
            if (!entry.is_regular_file()) continue;

            std::println("Processing file: {}", entry.path().c_str());

            TokenData& token_data = m_token_list.emplace_back(entry.path().string());

            std::ifstream file{entry.path()};
            std::string line;
            while (std::getline(file, line)) {
                for (const auto& token : tokenize(line)) {

                    const auto it = m_token_to_id.find(token);
                    if (it == m_token_to_id.end()) {
                        const auto new_id = next_id++;
                        std::println("new token: {} with ID: {}", token, new_id);
                        m_token_to_id.emplace(token, new_id);

                        token_data.add(new_id);
                    } else {
                        token_data.add(it->second);
                        std::println("Found token: {} with ID: {}", token, it->second);
                    }
                }
            }
        }
    }

}
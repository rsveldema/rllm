#pragma once

#include <NeuralNetwork.hpp>

#include <string>

namespace rllm
{
    class RLLM
    {
      public:
        RLLM(const std::vector<std::string>& filters);
        ~RLLM() = default;
        RLLM(const RLLM&) = delete;
        RLLM& operator=(const RLLM&) = delete;

        void train_mode(const std::string& filename, size_t num_layers, bool verbose,
                        TrainingMethod method, int window_size, size_t num_epochs);
        void prompt_mode(const std::string& filename);

      private:
        const std::vector<std::string> m_filters;
    };

} // namespace rllm
#pragma once

#include <NeuralNetwork.hpp>

#include <string>

namespace rllm
{
    class RLLM
    {
      public:
        RLLM();
        ~RLLM() = default;
        RLLM(const RLLM&) = delete;
        RLLM& operator=(const RLLM&) = delete;

        void train_mode(const std::string& filename, size_t num_layers, bool verbose);
        void prompt_mode(const std::string& filename);
    };

} // namespace rllm
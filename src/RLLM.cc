#include <RLLM.hpp>

#include <iostream>
#include <string>

namespace rllm
{
    RLLM::RLLM()
    {
        // Constructor implementation
    }

    void RLLM::prompt_mode()
    {
        std::string line;
        while (true) {
            std::cout << "Enter input (or 'exit' to quit): ";
            if (!std::getline(std::cin, line) || line == "exit") {
                break;
            }
            // Process the input line
        }
    }

    void RLLM::train_mode()
    {
        // Training logic implementation
    }
} // namespace rllm
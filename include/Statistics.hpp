#pragma once

#include <print>
#include <cstdint>
#include <string>

namespace rllm
{
    class Statistics
    {
      public:
        void record_learning_failure()
        {
            m_learning_failures++;
        }

        void record_learning_success()
        {
            m_learning_successes++;
        }

        void print_statistics() const
        {
            std::println("Total learning failures: {}", m_learning_failures);
            std::println("Total learning successes: {}", m_learning_successes);
        }

        private:
        size_t m_learning_failures = 0;
        size_t m_learning_successes = 0;
    };
} // namespace rllm
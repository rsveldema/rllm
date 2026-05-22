#pragma once

#include <chrono>
#include <cstdint>
#include <print>
#include <atomic>
#include <string>

namespace rllm
{


    class Statistics
    {
      public:
        class TotalLearnRecorderScope
        {
          public:
            TotalLearnRecorderScope(Statistics& stats)
                : m_stats(stats)
            {
                start  = std::chrono::steady_clock::now();
            }

            ~TotalLearnRecorderScope()
            {
                const auto end = std::chrono::steady_clock::now();
                const auto sum = end - start;

                m_stats.total_learning_duration.fetch_add(std::chrono::duration_cast<std::chrono::milliseconds>(sum).count());
            }

          private:
            Statistics& m_stats;
            std::chrono::steady_clock::time_point start;
        };


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

            auto duration = total_learning_duration.load();
            std::println("Total learning process took {} ms", duration);
        }

      private:
        size_t m_learning_failures = 0;
        size_t m_learning_successes = 0;
        std::atomic<size_t> total_learning_duration{0};
    };
} // namespace rllm
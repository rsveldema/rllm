#pragma once

#include <chrono>
#include <cstdint>
#include <print>
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
                    m_stats.record_start_learning_process();
                }

                ~TotalLearnRecorderScope()
                {
                    m_stats.record_end_learning_process();
                }

            private:
                Statistics& m_stats;
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

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                total_learning_end_time - total_learning_start_time
            )
                                .count();
            std::println("Total learning process took {} ms", duration);
        }


      private:
        size_t m_learning_failures = 0;
        size_t m_learning_successes = 0;

        std::chrono::steady_clock::time_point total_learning_start_time;
        std::chrono::steady_clock::time_point total_learning_end_time;

        void record_start_learning_process()
        {
            total_learning_start_time = std::chrono::steady_clock::now();
        }

        void record_end_learning_process()
        {
            total_learning_end_time = std::chrono::steady_clock::now();
        }

    };
} // namespace rllm
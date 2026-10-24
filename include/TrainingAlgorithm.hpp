#pragma once

#include <string_view>

namespace rllm
{
    enum class TrainingAlgorithm { Backprop, Dust };

    constexpr std::string_view training_algorithm_name(TrainingAlgorithm algorithm)
    {
        return algorithm == TrainingAlgorithm::Dust ? "dust" : "backprop";
    }
}

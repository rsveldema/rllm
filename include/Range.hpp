#pragma once

namespace rllm
{
    template <typename T>
    struct Range
    {
        T lo = T{0};
        T hi = T{1};
    };
} // namespace rllm

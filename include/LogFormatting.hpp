#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <cmath>
#include <format>

namespace rllm
{
    inline std::string escape_whitespace_for_log(std::string_view text)
    {
        std::string escaped;
        escaped.reserve(text.size());
        for (const char ch : text)
        {
            if (ch == '\n')
                escaped += "\\n";
            else if (ch == '\t')
                escaped += "\\t";
            else
                escaped += ch;
        }
        return escaped;
    }

    inline std::string format_eta_for_log(double seconds)
    {
        const auto total_seconds = static_cast<long long>(std::max(0.0, std::ceil(seconds)));
        const auto hours = total_seconds / 3600;
        const auto minutes = (total_seconds % 3600) / 60;
        const auto remaining_seconds = total_seconds % 60;
        return std::format("{:02}:{:02}:{:02}", hours, minutes, remaining_seconds);
    }
}

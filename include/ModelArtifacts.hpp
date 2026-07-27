#pragma once

#include <cstdlib>
#include <filesystem>
#include <string_view>

namespace rllm
{
    inline std::filesystem::path model_artifact_directory()
    {
        const char* configured_directory = std::getenv("RLLM_MODEL_DIR");
        if (configured_directory != nullptr && configured_directory[0] != '\0')
            return configured_directory;
        return "models";
    }

    inline std::filesystem::path model_artifact_path(std::string_view filename)
    {
        return model_artifact_directory() / filename;
    }
}

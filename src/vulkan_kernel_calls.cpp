#include <vulkan_kernel_calls.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <print>
#include <regex>
#include <string>
#include <vector>

#include <parallel.hpp>

namespace rllm::vulkan
{
    namespace detail
    {
        uint32_t count_ssbo_bindings_in_glsl(const std::filesystem::path& spirv_path)
        {
            std::filesystem::path comp_path = spirv_path;
            comp_path.replace_extension(".comp");

            std::ifstream input(comp_path);
            if (!input)
            {
                return 0;
            }

            const std::regex binding_re(R"(layout\s*\(\s*std430\s*,\s*set\s*=\s*0\s*,\s*binding\s*=\s*([0-9]+)\s*\))");
            uint32_t max_binding = 0;
            bool found = false;
            std::string line;
            while (std::getline(input, line))
            {
                std::smatch m;
                if (std::regex_search(line, m, binding_re))
                {
                    found = true;
                    uint32_t b = static_cast<uint32_t>(std::stoul(m[1].str()));
                    if (b > max_binding)
                    {
                        max_binding = b;
                    }
                }
            }

            return found ? (max_binding + 1u) : 0u;
        }

        detail::LocalSize parse_local_size_from_glsl(const std::filesystem::path& spirv_path)
        {
            std::filesystem::path comp_path = spirv_path;
            comp_path.replace_extension(".comp");

            std::ifstream input(comp_path);
            if (!input)
            {
                return {};
            }

            const std::regex local_size_re(R"(layout\s*\(\s*local_size_x\s*=\s*([0-9]+)\s*,\s*local_size_y\s*=\s*([0-9]+)\s*,\s*local_size_z\s*=\s*([0-9]+)\s*\)\s*in\s*;)");
            std::string line;
            while (std::getline(input, line))
            {
                std::smatch match;
                if (std::regex_search(line, match, local_size_re))
                {
                    return detail::LocalSize{
                        .x = static_cast<uint32_t>(std::stoul(match[1].str())),
                        .y = static_cast<uint32_t>(std::stoul(match[2].str())),
                        .z = static_cast<uint32_t>(std::stoul(match[3].str())),
                    };
                }
            }

            return {};
        }

        std::vector<uint32_t> load_spirv_words(const std::filesystem::path& spirv, const char* kernel_name)
        {
            std::ifstream input(spirv, std::ios::binary | std::ios::ate);
            if (!input)
            {
                LOG_ERROR("Failed to open SPIR-V kernel '{}' for kernel '{}' reading.", spirv.string(), kernel_name);
                std::abort();
            }

            const std::streamoff bytes = input.tellg();
            if (bytes <= 0 || (bytes % static_cast<std::streamoff>(sizeof(uint32_t))) != 0)
            {
                LOG_ERROR("SPIR-V kernel '{}' for kernel '{}' has invalid byte size {}.", spirv.string(), kernel_name, static_cast<long long>(bytes));
                std::abort();
            }

            input.seekg(0, std::ios::beg);
            std::vector<uint32_t> words(static_cast<size_t>(bytes) / sizeof(uint32_t));
            if (!input.read(reinterpret_cast<char*>(words.data()), bytes))
            {
                LOG_ERROR("Failed to read SPIR-V kernel '{}' bytes for kernel '{}'.", spirv.string(), kernel_name);
                std::abort();
            }
            return words;
        }

    } // namespace detail


    ComputeKernelRuntime::ComputeKernelRuntime(VulkanMemorySpace& space, std::string_view kernel_name, std::filesystem::path spirv_path)
        : m_space(space)
        , m_name(kernel_name)
        , m_spirv_path(std::move(spirv_path))
    {
        if (m_spirv_path.is_relative())
        {
            m_spirv_path = std::filesystem::path(RLLM_VULKAN_KERNEL_ROOT) / m_spirv_path;
        }
        m_spirv_words = detail::load_spirv_words(m_spirv_path, m_name.c_str());
        m_parsed_ssbo_binding_count = detail::count_ssbo_bindings_in_glsl(m_spirv_path);
        m_local_size = detail::parse_local_size_from_glsl(m_spirv_path);


        m_binding_count = std::max<uint32_t>(1u, m_parsed_ssbo_binding_count);
        m_bindings = std::vector<VkDescriptorSetLayoutBinding>(m_binding_count);

        for (uint32_t i = 0; i < static_cast<uint32_t>(m_binding_count); ++i)
        {
            m_bindings[i].binding = i;
            m_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            m_bindings[i].descriptorCount = 1;
            m_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            m_bindings[i].pImmutableSamplers = nullptr;
        }

        m_buffer_infos = std::vector<VkDescriptorBufferInfo>(m_binding_count);

        m_writes = std::vector<VkWriteDescriptorSet>(m_binding_count);
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_binding_count); ++i)
        {
            m_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            m_writes[i].dstSet = VK_NULL_HANDLE;
            m_writes[i].dstBinding = i;
            m_writes[i].dstArrayElement = 0;
            m_writes[i].descriptorCount = 1;
            m_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            m_writes[i].pBufferInfo = nullptr;
        }

    }

} // namespace rllm::vulkan
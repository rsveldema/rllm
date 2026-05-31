#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <type_traits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <logging.hpp>
#include <vulkan/vulkan.h>

namespace rllm::vulkan
{
#ifndef RLLM_VULKAN_KERNEL_ROOT
#define RLLM_VULKAN_KERNEL_ROOT "generated/vulkanized/kernels"
#endif

    namespace detail
    {
        struct HostBufferView
        {
            void* host_ptr = nullptr;
            size_t size_bytes = 0;
            bool writable = false;
        };

        template <typename T>
        inline constexpr bool is_host_buffer_arg_v =
            requires(T value) {
                value.data();
                value.storage_size_bytes();
            };

        template <typename Arg>
        inline void append_buffer_view(std::vector<HostBufferView>& out, Arg&& arg)
        {
            using ArgType = std::remove_reference_t<Arg>;
            if constexpr (is_host_buffer_arg_v<ArgType>)
            {
                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(arg.data()));
                view.size_bytes = static_cast<size_t>(arg.storage_size_bytes());
                view.writable = !std::is_const_v<ArgType>;
                out.push_back(view);
            }
            else
            {
                static_cast<void>(out);
                static_cast<void>(arg);
            }
        }

        std::vector<uint32_t> load_spirv_words(const std::filesystem::path& spirv, const char* kernel_name);

        uint32_t count_ssbo_bindings_in_glsl(const std::filesystem::path& spirv_path);
    }

        class ComputeKernel
        {
            public:
                ComputeKernel(std::string_view kernel_name, std::filesystem::path spirv_path);

                ~ComputeKernel();

                ComputeKernel(const ComputeKernel&) = delete;
                ComputeKernel& operator=(const ComputeKernel&) = delete;
                ComputeKernel(ComputeKernel&&) = delete;
                ComputeKernel& operator=(ComputeKernel&&) = delete;

                const std::string& name() const { return m_name; }
                const std::filesystem::path& spirv_path() const { return m_spirv_path; }
                const std::vector<uint32_t>& spirv_words() const { return m_spirv_words; }
                VkPipeline pipeline() const { return m_pipeline; }
                template <typename Range, typename... Args>
                void launch_1d(Range&& range, Args&&... args);
                template <typename Range2D, typename... Args>
                void launch_2d(Range2D&& range, Args&&... args);
                void dispatch_kernel(uint32_t groups_x, uint32_t groups_y, const std::vector<detail::HostBufferView>& buffers);
                void ensure_pipeline(VkDevice device, uint32_t ssbo_binding_count);

            private:
                std::string m_name;
                std::filesystem::path m_spirv_path;
                std::vector<uint32_t> m_spirv_words;
                VkDevice m_cached_device = VK_NULL_HANDLE;
                VkShaderModule m_shader_module = VK_NULL_HANDLE;
                VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
                VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
                VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
                VkDescriptorSet m_descriptor_set = VK_NULL_HANDLE;
                VkPipeline m_pipeline = VK_NULL_HANDLE;
                uint32_t m_ssbo_binding_count = 0;
                VkDevice m_command_buffer_device = VK_NULL_HANDLE;
                VkCommandPool m_command_pool = VK_NULL_HANDLE;
                VkCommandBuffer m_command_buffer = VK_NULL_HANDLE;
        };

    namespace detail
    {
        struct RuntimeContext
        {
            VkInstance instance = VK_NULL_HANDLE;
            VkPhysicalDevice physical_device = VK_NULL_HANDLE;
            VkDevice device = VK_NULL_HANDLE;
            uint32_t queue_family_index = 0;
            VkQueue queue = VK_NULL_HANDLE;
            VkCommandPool command_pool = VK_NULL_HANDLE;
        };

        inline uint32_t ceil_div_u32(uint32_t v, uint32_t d)
        {
            return (v + d - 1u) / d;
        }

        template <typename T>
        inline size_t to_size(T value)
        {
            return static_cast<size_t>(value);
        }

        template <typename Range>
        inline uint32_t range_size_1d(const Range& range)
        {
            size_t count = 0;
            if constexpr (requires { range.size(); })
            {
                count = to_size(range.size());
            }
            else if constexpr (requires { range.hi; range.lo; })
            {
                count = to_size(range.hi - range.lo);
            }
            else if constexpr (requires { range.end(); range.begin(); })
            {
                count = to_size(range.end() - range.begin());
            }
            else
            {
                LOG_ERROR("Unable to infer 1D dispatch size for Vulkan kernel range type.");
                std::abort();
            }

            if (count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
            {
                LOG_ERROR("Vulkan dispatch size {} exceeds uint32_t limits.", count);
                std::abort();
            }
            return static_cast<uint32_t>(count);
        }

        template <typename Range2D>
        inline std::pair<uint32_t, uint32_t> range_size_2d(const Range2D& range)
        {
            size_t x = 0;
            size_t y = 0;

            if constexpr (requires { range.inner_size(); range.outer_size(); })
            {
                x = to_size(range.inner_size());
                y = to_size(range.outer_size());
            }
            else if constexpr (requires { range.cols(); range.rows(); })
            {
                x = to_size(range.cols());
                y = to_size(range.rows());
            }
            else
            {
                x = to_size(range_size_1d(range));
                y = 1;
            }

            if (x > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                y > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
            {
                LOG_ERROR("Vulkan 2D dispatch size ({}, {}) exceeds uint32_t limits.", x, y);
                std::abort();
            }

            return {static_cast<uint32_t>(x), static_cast<uint32_t>(y)};
        }

        RuntimeContext create_runtime_context();

        RuntimeContext& runtime_context();

    } // namespace detail

    template <typename Range, typename... Args>
    inline void ComputeKernel::launch_1d(Range&& range, Args&&... args)
    {
        if (!std::filesystem::exists(spirv_path()))
        {
            LOG_ERROR(
                "Missing generated Vulkan kernel artifact '{}' for kernel '{}'.",
                spirv_path().string(),
                name()
            );
            std::abort();
        }

        constexpr uint32_t kLocalSizeX = 64;
        const uint32_t x_items = detail::range_size_1d(range);
        const uint32_t groups_x = detail::ceil_div_u32(x_items, kLocalSizeX);

        std::vector<detail::HostBufferView> buffers;
        buffers.reserve(sizeof...(Args));
        (detail::append_buffer_view(buffers, std::forward<Args>(args)), ...);

        dispatch_kernel(groups_x, 1, buffers);
    }

    template <typename Range2D, typename... Args>
    inline void ComputeKernel::launch_2d(Range2D&& range, Args&&... args)
    {
        if (!std::filesystem::exists(spirv_path()))
        {
            LOG_ERROR(
                "Missing generated Vulkan kernel artifact '{}' for kernel '{}'.",
                spirv_path().string(),
                name()
            );
            std::abort();
        }

        constexpr uint32_t kLocalSizeX = 64;
        constexpr uint32_t kLocalSizeY = 1;
        const auto [x_items, y_items] = detail::range_size_2d(range);
        const uint32_t groups_x = detail::ceil_div_u32(x_items, kLocalSizeX);
        const uint32_t groups_y = detail::ceil_div_u32(y_items, kLocalSizeY);

        std::vector<detail::HostBufferView> buffers;
        buffers.reserve(sizeof...(Args));
        (detail::append_buffer_view(buffers, std::forward<Args>(args)), ...);

        dispatch_kernel(groups_x, groups_y, buffers);
    }

} // namespace rllm::vulkan
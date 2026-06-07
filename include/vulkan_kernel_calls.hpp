#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <device_pointer.hpp>
#include <logging.hpp>
#include <parallel.hpp>
#include <vulkan/vulkan.h>


#ifndef RLLM_VULKAN_KERNEL_ROOT
#define RLLM_VULKAN_KERNEL_ROOT "generated/vulkanized/kernels"
#endif

namespace rllm::vulkan
{
    namespace detail
    {

        struct LocalSize
        {
            uint32_t x = 1;
            uint32_t y = 1;
            uint32_t z = 1;
        };
    }


    class ComputeKernelRuntime
    {
public:
        ComputeKernelRuntime(VulkanMemorySpace& space, std::string_view kernel_name, std::filesystem::path spirv_path)
            : m_space(space), m_name(kernel_name), m_spirv_path(spirv_path)
        {
        }

        ~ComputeKernelRuntime() {}

        ComputeKernelRuntime(const ComputeKernelRuntime&) = delete;
        ComputeKernelRuntime& operator=(const ComputeKernelRuntime&) = delete;
        ComputeKernelRuntime(ComputeKernelRuntime&&) = delete;
        ComputeKernelRuntime& operator=(ComputeKernelRuntime&&) = delete;
        

        const std::string& name() const
        {
            return m_name;
        }
        const std::filesystem::path& spirv_path() const
        {
            return m_spirv_path;
        }
        const std::vector<uint32_t>& spirv_words() const
        {
            return m_spirv_words;
        }
        VkPipeline pipeline() const
        {
            return m_pipeline;
        }
        uint32_t local_size_x() const
        {
            return m_local_size.x;
        }
        uint32_t local_size_y() const
        {
            return m_local_size.y;
        }
        uint32_t local_size_z() const
        {
            return m_local_size.z;
        }

        struct RuntimeBuffer
        {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VkDeviceMemory staging_memory = VK_NULL_HANDLE;
            void* mapped = nullptr;
            bool cached = false;  // true when buffer is kept alive for lazy-flush DevicePointer args
            bool use_offload_source = false;
            bool bind_offload_direct = false;
            VkDeviceSize offload_source_offset = 0;
        };

        VulkanMemorySpace& m_space;

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
        uint32_t m_parsed_ssbo_binding_count = 0;
        detail::LocalSize m_local_size{};
        VkDevice m_command_buffer_device = VK_NULL_HANDLE;
        VkCommandPool m_command_pool = VK_NULL_HANDLE;
        VkCommandBuffer m_command_buffer = VK_NULL_HANDLE;
        VkDevice m_submit_fence_device = VK_NULL_HANDLE;
        VkFence m_submit_fence = VK_NULL_HANDLE;
        VkDevice m_timestamp_query_pool_device = VK_NULL_HANDLE;
        VkQueryPool m_timestamp_query_pool = VK_NULL_HANDLE;
        float m_timestamp_period_ns = 0.0f;
        uint32_t m_timestamp_valid_bits = 0;
        bool m_pending_submit_has_timestamps = false;
        bool m_submit_in_flight = false;
        std::string m_pending_submit_kernel_name;  // kernel name for the in-flight submit being waited on
        std::mutex m_launch_mutex;
    };

    template <typename... KernelArgs>
    class ComputeKernel : public ComputeKernelRuntime
    {
      public:         
        using ComputeKernelRuntime::ComputeKernelRuntime;

        template <typename Range>
        void launch_1d_named(Range&& range, std::initializer_list<std::string_view> parameter_names, KernelArgs... args)
        {}

        template <typename Range2D>
        void launch_2d_named(Range2D&& range, std::initializer_list<std::string_view> parameter_names, KernelArgs... args)
        {}

        template <typename Range3D>
        void launch_3d_named(Range3D&& range, std::initializer_list<std::string_view> parameter_names, KernelArgs... args)
        {}
    };
} // namespace rllm::vulkan

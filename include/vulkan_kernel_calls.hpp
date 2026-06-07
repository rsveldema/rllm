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
        ComputeKernelRuntime(VulkanMemorySpace& space, std::string_view kernel_name, std::filesystem::path spirv_path);

        ~ComputeKernelRuntime() {
            // can be left empty, instances are statically allocated
            // and will be removed at program exit anyway.
        }

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

        VulkanMemorySpace& m_space;

        std::string m_name;
        std::filesystem::path m_spirv_path;
        std::vector<uint32_t> m_spirv_words;
        uint32_t m_parsed_ssbo_binding_count = 0;
        detail::LocalSize m_local_size{};
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

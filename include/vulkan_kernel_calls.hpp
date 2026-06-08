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

#include <cassert>

#ifndef RLLM_VULKAN_KERNEL_ROOT
#define RLLM_VULKAN_KERNEL_ROOT "generated/vulkanized/kernels"
#endif

namespace rllm::vulkan
{
    template <typename T>
    inline constexpr bool is_scalar_kernel_arg_v = (std::is_enum_v<std::remove_cv_t<std::remove_reference_t<T>>> || std::is_integral_v<std::remove_cv_t<std::remove_reference_t<T>>> || std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<T>>>);
    template <typename TupleType, size_t I>
    struct count_non_scalars_before
    {
        static constexpr size_t value = (is_scalar_kernel_arg_v<std::remove_cv_t<std::tuple_element_t<I-1, TupleType>>> ? 0 : 1) + count_non_scalars_before<TupleType, I-1>::value;
    };
    template <typename TupleType>
    struct count_non_scalars_before<TupleType, 0>
    {
        static constexpr size_t value = 0;
    };

    namespace detail
    {
        struct LocalSize
        {
            uint32_t x = 1, y = 1, z = 1;
        };

        template <typename Arg>
        inline void append_scalar_arg(std::vector<uint32_t>& out, size_t& out_size, Arg&& arg)
        {
            using ArgType = std::remove_cv_t<std::remove_reference_t<Arg>>;
            if constexpr (is_scalar_kernel_arg_v<ArgType>)
            {
                if (out_size >= out.size())
                {
                    LOG_ERROR("Vulkan launch scalar buffer overflow");
                    std::abort();
                }
                const auto value = arg;
                std::memcpy(&out[out_size], &value, sizeof(value));
                out_size++;
            }
            else
            {
                static_cast<void>(out);
                static_cast<void>(out_size);
                static_cast<void>(arg);
            }

        }
    } // namespace detail

    // Helper functions for Vulkan kernel argument setup using index_sequence.

    struct buf_helper { size_t idx = 0; };

    template <size_t I>
    void helper_set_one_buffer(std::vector<VkDescriptorBufferInfo>& binfos,
                               const std::vector<std::string_view>& pn, auto&& t)
    {
        using ArgType = std::tuple_element_t<I, std::decay_t<decltype(t)>>;
        if constexpr (!is_scalar_kernel_arg_v<std::remove_cv_t<std::remove_reference_t<ArgType>>>)
        {
            auto& ref = std::get<I>(t);
            ::parallel::set_vulkan_dispatch_params(pn[I], pn[I]);
            VkBuffer buf;
            if constexpr (requires(const ArgType& x) { x.raw_offload_data(); })
                buf = ref.raw_offload_data().get();
            else
                buf = ref.offload_data().get();
            binfos[count_non_scalars_before<std::decay_t<decltype(t)>, I>::value].buffer = buf;
            ::parallel::clear_vulkan_dispatch_params();
        }
    }

    template <size_t I>
    void helper_append_one_scalar(std::vector<uint32_t>& out_bytes, size_t& out_size, auto&& t)
    {
        using ArgType = std::tuple_element_t<I, std::decay_t<decltype(t)>>;
        if constexpr (is_scalar_kernel_arg_v<std::remove_cv_t<std::remove_reference_t<ArgType>>>)
            detail::append_scalar_arg(out_bytes, out_size, std::get<I>(t));
    }

    template <typename ArgTupleType, size_t... Is>
    void helper_set_buffer_descriptors(std::vector<VkDescriptorBufferInfo>& binfos,
                                       std::initializer_list<std::string_view> pn,
                                       ArgTupleType&& t,
                                       std::index_sequence<Is...>)
    {
        (helper_set_one_buffer<Is>(binfos, pn, std::forward<ArgTupleType>(t)), ...);
    }

    template <typename ArgTupleType, size_t... Is>
    void helper_append_scalar_args(std::vector<uint32_t>& out_bytes, size_t& out_size,
                                   ArgTupleType&& t, std::index_sequence<Is...>)
    {
        (helper_append_one_scalar<Is>(out_bytes, out_size, std::forward<ArgTupleType>(t)), ...);
    }

    class ComputeKernelRuntime
    {
      public:
        ComputeKernelRuntime(VulkanMemorySpace& space, std::string_view kernel_name, std::filesystem::path spirv_path, size_t num_buffer_args, size_t num_constant_args)
            : m_space(space)
            , m_name(kernel_name)
            , m_spirv_path(std::move(spirv_path))
        {
            space.register_kernel(this);

            if (m_spirv_path.is_relative())
                m_spirv_path = std::filesystem::path(RLLM_VULKAN_KERNEL_ROOT) / m_spirv_path;
            std::ifstream input(m_spirv_path, std::ios::binary | std::ios::ate);
            if (!input)
            {
                LOG_ERROR("Failed to open SPIR-V kernel for reading");
                std::abort();
            }
            const auto bytes = input.tellg();
            m_spirv_words.resize(static_cast<size_t>(bytes) / sizeof(uint32_t));
            input.seekg(0, std::ios::beg);
            if (!input.read(reinterpret_cast<char*>(m_spirv_words.data()), bytes))
            {
                LOG_ERROR("Failed to read SPIR-V");
                std::abort();
            }
            m_binding_count = static_cast<uint32_t>(std::max(static_cast<size_t>(1), num_buffer_args));
            m_bindings.resize(m_binding_count);
            m_buffer_infos.resize(m_binding_count);
            m_writes.resize(m_binding_count);
            for (uint32_t i = 0; i < m_binding_count; ++i)
            {
                m_bindings[i].binding = i;
                m_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                m_bindings[i].descriptorCount = 1;
                m_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                m_buffer_infos[i] = VkDescriptorBufferInfo{};
                m_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                m_writes[i].dstBinding = i;
                m_writes[i].descriptorCount = 1;
                m_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

            }
            m_push_constant_bytes.resize(num_constant_args);
        }
        ~ComputeKernelRuntime()
        {
            VkDevice dev = m_space.device();
            if (m_command_buffer != VK_NULL_HANDLE)
                vkFreeCommandBuffers(dev, m_space.command_pool(), 1, &m_command_buffer);
            if (m_descriptor_pool != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(dev, m_descriptor_pool, nullptr);
            if (m_pipeline != VK_NULL_HANDLE)
                vkDestroyPipeline(dev, m_pipeline, nullptr);
            if (m_pipeline_layout != VK_NULL_HANDLE)
                vkDestroyPipelineLayout(dev, m_pipeline_layout, nullptr);
            if (m_descriptor_set_layout != VK_NULL_HANDLE)
                vkDestroyDescriptorSetLayout(dev, m_descriptor_set_layout, nullptr);
            if (m_shader_module != VK_NULL_HANDLE)
                vkDestroyShaderModule(dev, m_shader_module, nullptr);
        }
        ComputeKernelRuntime(const ComputeKernelRuntime&) = delete;
        ComputeKernelRuntime& operator=(const ComputeKernelRuntime&) = delete;
        ComputeKernelRuntime(ComputeKernelRuntime&&) = delete;
        ComputeKernelRuntime& operator=(ComputeKernelRuntime&&) = delete;

        void set_id(size_t unique_id) {
            unique_id = unique_id;
        }

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
      
      protected:
        size_t unique_id = 0;
        VulkanMemorySpace& m_space;
        std::string m_name;
        std::filesystem::path m_spirv_path;
        std::vector<uint32_t> m_spirv_words;
        detail::LocalSize m_local_size{};
        VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet m_descriptor_set = VK_NULL_HANDLE;
        VkPipeline m_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
        VkShaderModule m_shader_module = VK_NULL_HANDLE;
        VkCommandBuffer m_command_buffer = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayoutBinding> m_bindings;
        std::vector<VkDescriptorBufferInfo> m_buffer_infos;
        std::vector<VkWriteDescriptorSet> m_writes;
        uint32_t m_binding_count = 0;
        std::vector<uint32_t> m_push_constant_bytes;
        size_t m_push_constant_size = 0;

      protected:
        void create_vulkan_compute_pipeline()
        {
            VkDevice dev = m_space.device();
            if (!m_shader_module)
            {
                VkShaderModuleCreateInfo smci{};
                smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                smci.codeSize = m_spirv_words.size() * sizeof(uint32_t);
                smci.pCode = m_spirv_words.data();
                vkCreateShaderModule(dev, &smci, nullptr, &m_shader_module);
            }
            if (!m_descriptor_set_layout)
            {
                VkDescriptorSetLayoutCreateInfo dslci{};
                dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                dslci.bindingCount = m_binding_count;
                assert(m_binding_count > 0 && static_cast<size_t>(m_binding_count) <= m_bindings.size());
                dslci.pBindings = m_bindings.data();
                vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &m_descriptor_set_layout);
            }
            if (!m_pipeline_layout)
            {
                VkPipelineLayoutCreateInfo plci{};
                plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                plci.setLayoutCount = 1;
                plci.pSetLayouts = &m_descriptor_set_layout;
                vkCreatePipelineLayout(dev, &plci, nullptr, &m_pipeline_layout);
            }
            if (!m_pipeline)
            {
                VkComputePipelineCreateInfo cpci{};
                cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                VkPipelineShaderStageCreateInfo pssci{};
                pssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                pssci.module = m_shader_module;
                pssci.pName = "main";
                cpci.stage = pssci;
                cpci.layout = m_pipeline_layout;
                vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline);
            }
            // Lazy-init descriptor pool + set for cached reuse
            if (!m_descriptor_pool)
            {
                VkDescriptorPoolSize pool_size{};
                pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                pool_size.descriptorCount = m_binding_count;
                VkDescriptorPoolCreateInfo dpci{};
                dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                dpci.maxSets = 1;
                dpci.poolSizeCount = 1;
                dpci.pPoolSizes = &pool_size;
                vkCreateDescriptorPool(dev, &dpci, nullptr, &m_descriptor_pool);
            }

            if (!m_descriptor_set)
            {
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = m_descriptor_pool;
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts = &m_descriptor_set_layout;
                vkAllocateDescriptorSets(dev, &dsai, &m_descriptor_set);

            }
            // Lazy-init command buffer for cached reuse
            if (!m_command_buffer)
            {
                VkCommandBufferAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                ai.commandPool = m_space.command_pool();
                ai.commandBufferCount = 1;
                vkAllocateCommandBuffers(dev, &ai, &m_command_buffer);
            }
        }

      public:
        template <typename KernelArgs, typename Range1D>
        void launch_1d(Range1D&& r, std::string_view n, const KernelArgs& a)
        {
            create_vulkan_compute_pipeline();
        }
        template <typename KernelArgs, typename Range2D>
        void launch_2d(Range2D&& r, const KernelArgs& a)
        {
            create_vulkan_compute_pipeline();
        }
        template <typename KernelArgs, typename Range3D>
        void launch_3d(Range3D&& r, const KernelArgs& a)
        {
            create_vulkan_compute_pipeline();
        }
    }; // end class


    template <typename... Args>
    struct ComputeKernel : public ComputeKernelRuntime
    {
        constexpr static size_t k_buffer_arg_count = (0 + ... + (is_scalar_kernel_arg_v<std::remove_cv_t<std::remove_reference_t<Args>>> ? 0 : 1));
        constexpr static size_t k_constant_arg_count = (0 + ... + (is_scalar_kernel_arg_v<std::remove_cv_t<std::remove_reference_t<Args>>> ? 1 : 0));
        ComputeKernel(VulkanMemorySpace& s, std::string_view n, std::filesystem::path p)
            : ComputeKernelRuntime(s, n, std::move(p), k_buffer_arg_count, k_constant_arg_count)
        {}

        template <typename Range2D>
        void dispatch_named(Range2D&& range, const uint32_t gx, const uint32_t gy, const uint32_t gz, std::initializer_list<std::string_view> pn_in, Args&&... args)
        {
            // Use pack expansion with index_sequence for compile-time constant indices.
            // Each Args[i] accessed via std::get<i>(tuple) where i is constexpr at each instantiation.
            // Assert that cached vectors are large enough for binding_count
            assert(static_cast<size_t>(m_binding_count) <= m_bindings.size());
            assert(static_cast<size_t>(m_binding_count) <= m_buffer_infos.size());
            assert(static_cast<size_t>(m_binding_count) <= m_writes.size());
            helper_set_buffer_descriptors(m_buffer_infos, pn_in,
                std::forward_as_tuple(std::forward<Args>(args)...),
                std::index_sequence_for<Args...>{});

            for (auto& b : m_buffer_infos)
            {
                b.offset = 0;
                b.range = VK_WHOLE_SIZE;
            }
            for (uint32_t i = 0; i < m_binding_count; ++i)
            {
                assert(static_cast<size_t>(i) < m_writes.size());
                assert(static_cast<size_t>(i) < m_buffer_infos.size());
                assert(m_descriptor_set != VK_NULL_HANDLE);
                m_writes[i].dstSet = m_descriptor_set;
                m_writes[i].pBufferInfo = &m_buffer_infos[i];
            }
            vkUpdateDescriptorSets(m_space.device(), m_binding_count, m_writes.data(), 0, nullptr);

            // Reset push constant size and clear bytes without reallocating
            if constexpr (k_constant_arg_count > 0)
            {
                m_push_constant_size = 0;
                helper_append_scalar_args(m_push_constant_bytes, m_push_constant_size,
                    std::forward_as_tuple(std::forward<Args>(args)...),
                    std::index_sequence_for<Args...>{});

            }

            // Use cached m_command_buffer (allocated once)
                VkCommandBufferBeginInfo bi{};
            vkResetCommandBuffer(m_command_buffer, 0);
                bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
                vkBeginCommandBuffer(m_command_buffer, &bi);
                vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
                vkCmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 0, 1, &m_descriptor_set, 0, nullptr);
                if (m_push_constant_size > 0)
                    assert(static_cast<size_t>(m_push_constant_size) <= m_push_constant_bytes.size());
                    vkCmdPushConstants(m_command_buffer, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, m_push_constant_size * sizeof(uint32_t), m_push_constant_bytes.data());
                vkCmdDispatch(m_command_buffer, gx, gy, gz);
                vkEndCommandBuffer(m_command_buffer);
                VkSubmitInfo si{};
                si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                si.commandBufferCount = 1;
                si.pCommandBuffers = &m_command_buffer;
                vkQueueSubmit(m_space.queue(), 1, &si, VK_NULL_HANDLE);
                vkQueueWaitIdle(m_space.queue());

        }

      private:
        template <typename A>
        static VkBuffer buf_from_arg(const A& a)
        {
            if constexpr (requires(const A& x) { x.raw_offload_data(); })
                return a.raw_offload_data().get();
            else
                return a.offload_data().get();
        }

      public:
        template <typename Range1D>
        void launch_1d_named(Range1D&& r, std::initializer_list<std::string_view> pn_in, Args&&... args)
        {
            const uint32_t gx = static_cast<uint32_t>((r.inner_size() + local_size_x() - 1u) / local_size_x());
            const uint32_t gy = 1;
            const uint32_t gz = 1;
            create_vulkan_compute_pipeline();
            dispatch_named(r, gx, gy, gz, pn_in, std::forward<Args>(args)...);
        }

        template <typename Range2D>
        void launch_2d_named(Range2D&& r, std::initializer_list<std::string_view> pn_in, Args&&... args)
        {
            const uint32_t gx = static_cast<uint32_t>((r.inner_size() + local_size_x() - 1u) / local_size_x());
            const uint32_t gy = static_cast<uint32_t>((r.outer_size() + local_size_y() - 1u) / local_size_y());
            const uint32_t gz = 1;
            create_vulkan_compute_pipeline();
            dispatch_named(r, gx, gy, gz, pn_in, std::forward<Args>(args)...);
        }

        template <typename Range3D>
        void launch_3d(Range3D&& r, std::initializer_list<std::string_view> pn_in, Args&&... args)
        {
            const uint32_t gx = static_cast<uint32_t>((r.inner_size() + local_size_x() - 1u) / local_size_x());
            const uint32_t gy = static_cast<uint32_t>((r.outer_size() + local_size_y() - 1u) / local_size_y());
            const uint32_t gz = static_cast<uint32_t>((r.z_size() + local_size_z() - 1u) / local_size_z());
            create_vulkan_compute_pipeline();
            dispatch_named(r, gx, gy, gz, pn_in, std::forward<Args>(args)...);
        }
        template <typename Range3D>
        void launch_3d_named(Range3D&& r, std::initializer_list<std::string_view> pn_in, Args&&... args)
        {
            const uint32_t gx = static_cast<uint32_t>((r.inner_size() + local_size_x() - 1u) / local_size_x());
            const uint32_t gy = static_cast<uint32_t>((r.outer_size() + local_size_y() - 1u) / local_size_y());
            const uint32_t gz = static_cast<uint32_t>((r.z_size() + local_size_z() - 1u) / local_size_z());
            create_vulkan_compute_pipeline();
            dispatch_named(r, gx, gy, gz, pn_in, std::forward<Args>(args)...);
        }
    };


} // namespace rllm::vulkan

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
        {
            // determine dispatch dimensions
            auto dims = []<typename R>(const R& r)
            {
                return std::tuple<size_t,size_t,size_t>{static_cast<size_t>(r.end() - r.begin()), 1u, 1u};
            }(range);

            const size_t total_x = std::get<0>(dims);
            if (total_x == 0)
                return;

            const uint32_t local_x = std::max<uint32_t>(1u, m_local_size.x);
            const uint32_t group_x = static_cast<uint32_t>((total_x + local_x - 1) / local_x);

            // forward to common dispatch helper
            dispatch_compute(parameter_names, group_x, 1u, 1u, args...);
        }

        template <typename Range2D>
        void launch_2d_named(Range2D&& range, std::initializer_list<std::string_view> parameter_names, KernelArgs... args)
        {
            size_t size_x = 1, size_y = 1;
            if constexpr (requires(const Range2D& r) { r.inner_size(); r.outer_size(); })
            {
                size_x = range.inner_size();
                size_y = range.outer_size();
            }
            else
            {
                const int tot = range.end() - range.begin();
                size_x = static_cast<size_t>(tot);
            }

            if (size_x == 0 || size_y == 0)
                return;

            const uint32_t local_x = std::max<uint32_t>(1u, m_local_size.x);
            const uint32_t local_y = std::max<uint32_t>(1u, m_local_size.y);
            const uint32_t gx = static_cast<uint32_t>((size_x + local_x - 1) / local_x);
            const uint32_t gy = static_cast<uint32_t>((size_y + local_y - 1) / local_y);

            dispatch_compute(parameter_names, gx, gy, 1u, args...);
        }

        template <typename Range3D>
        void launch_3d_named(Range3D&& range, std::initializer_list<std::string_view> parameter_names, KernelArgs... args)
        {
            size_t ix = 1, iy = 1, iz = 1;
            if constexpr (requires(const Range3D& r) { r.inner_size(); r.middle_size(); r.outer_size(); })
            {
                ix = range.inner_size();
                iy = range.middle_size();
                iz = range.outer_size();
            }
            else if constexpr (requires(const Range3D& r) { r.inner_size(); r.outer_size(); })
            {
                ix = range.inner_size();
                iy = range.outer_size();
            }
            else
            {
                const int tot = range.end() - range.begin();
                ix = static_cast<size_t>(tot);
            }

            if (ix == 0 || iy == 0 || iz == 0)
                return;

            const uint32_t lx = std::max<uint32_t>(1u, m_local_size.x);
            const uint32_t ly = std::max<uint32_t>(1u, m_local_size.y);
            const uint32_t lz = std::max<uint32_t>(1u, m_local_size.z);
            const uint32_t gx = static_cast<uint32_t>((ix + lx - 1) / lx);
            const uint32_t gy = static_cast<uint32_t>((iy + ly - 1) / ly);
            const uint32_t gz = static_cast<uint32_t>((iz + lz - 1) / lz);

            dispatch_compute(parameter_names, gx, gy, gz, args...);
        }

      private:
        // Helper to extract a Vulkan buffer from a kernel argument
        template <typename A>
        static VkBuffer buffer_from_arg(const A& a)
        {
            if constexpr (requires(const A& x) { x.raw_offload_data(); })
            {
                return a.raw_offload_data().get();
            }
            else if constexpr (requires(const A& x) { x.offload_data(); })
            {
                return a.offload_data().get();
            }
            else
            {
                static_assert(sizeof(A) == 0, "Kernel argument does not provide offload buffer accessor");
            }
        }

        // Core dispatch implementation: creates shader, pipeline, descriptor sets, and dispatches.
        template <typename... Args>
        void dispatch_compute(std::initializer_list<std::string_view> parameter_names, uint32_t gx, uint32_t gy, uint32_t gz, Args&&... args)
        {
            // create shader module
            VkShaderModuleCreateInfo smci{};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = m_spirv_words.size() * sizeof(uint32_t);
            smci.pCode = m_spirv_words.data();
            VkShaderModule shader_module = VK_NULL_HANDLE;
            VkResult r = vkCreateShaderModule(m_space.device(), &smci, nullptr, &shader_module);
            if (r != VK_SUCCESS)
            {
                LOG_ERROR("vkCreateShaderModule failed: {}", static_cast<int>(r));
                std::abort();
            }

            // descriptor layout
            const uint32_t binding_count = std::max<uint32_t>(1u, m_parsed_ssbo_binding_count);
            std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
            for (uint32_t i = 0; i < binding_count; ++i)
            {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                bindings[i].pImmutableSamplers = nullptr;
            }

            VkDescriptorSetLayoutCreateInfo dslci{};
            dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dslci.bindingCount = static_cast<uint32_t>(bindings.size());
            dslci.pBindings = bindings.data();
            VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
            r = vkCreateDescriptorSetLayout(m_space.device(), &dslci, nullptr, &descriptor_set_layout);
            if (r != VK_SUCCESS)
            {
                LOG_ERROR("vkCreateDescriptorSetLayout failed: {}", static_cast<int>(r));
                std::abort();
            }

            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &descriptor_set_layout;
            VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
            r = vkCreatePipelineLayout(m_space.device(), &plci, nullptr, &pipeline_layout);
            if (r != VK_SUCCESS)
            {
                LOG_ERROR("vkCreatePipelineLayout failed: {}", static_cast<int>(r));
                std::abort();
            }

            VkPipeline pipeline = VK_NULL_HANDLE;
            VkComputePipelineCreateInfo cpci{};
            cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            VkPipelineShaderStageCreateInfo pssci{};
            pssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pssci.module = shader_module;
            pssci.pName = "main";
            cpci.stage = pssci;
            cpci.layout = pipeline_layout;

            r = vkCreateComputePipelines(m_space.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);
            if (r != VK_SUCCESS)
            {
                LOG_ERROR("vkCreateComputePipelines failed: {}", static_cast<int>(r));
                std::abort();
            }

            // descriptor pool + set
            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pool_size.descriptorCount = binding_count;

            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets = 1;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &pool_size;
            VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
            r = vkCreateDescriptorPool(m_space.device(), &dpci, nullptr, &descriptor_pool);
            if (r != VK_SUCCESS)
            {
                LOG_ERROR("vkCreateDescriptorPool failed: {}", static_cast<int>(r));
                std::abort();
            }

            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool = descriptor_pool;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &descriptor_set_layout;
            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
            r = vkAllocateDescriptorSets(m_space.device(), &dsai, &descriptor_set);
            if (r != VK_SUCCESS)
            {
                LOG_ERROR("vkAllocateDescriptorSets failed: {}", static_cast<int>(r));
                std::abort();
            }

            // prepare buffer infos from args (map by order)
            std::vector<VkDescriptorBufferInfo> buffer_infos(binding_count);
            {
                size_t idx = 0;
                (void)std::initializer_list<int>{([&]{ if (idx < buffer_infos.size()) buffer_infos[idx++].buffer = buffer_from_arg(args); return 0; }(), 0)...};
            }
            for (auto &bi : buffer_infos)
            {
                bi.offset = 0;
                bi.range = VK_WHOLE_SIZE;
            }

            std::vector<VkWriteDescriptorSet> writes(binding_count);
            for (uint32_t i = 0; i < binding_count; ++i)
            {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = descriptor_set;
                writes[i].dstBinding = i;
                writes[i].dstArrayElement = 0;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &buffer_infos[i];
            }
            vkUpdateDescriptorSets(m_space.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            // record and submit command buffer
            VkCommandBuffer cmd = m_space.begin_one_time_command();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
            vkCmdDispatch(cmd, gx, gy, gz);
            m_space.end_one_time_command(cmd);

            // cleanup
            vkDestroyDescriptorPool(m_space.device(), descriptor_pool, nullptr);
            vkDestroyPipeline(m_space.device(), pipeline, nullptr);
            vkDestroyPipelineLayout(m_space.device(), pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(m_space.device(), descriptor_set_layout, nullptr);
            vkDestroyShaderModule(m_space.device(), shader_module, nullptr);
        }
    };
} // namespace rllm::vulkan

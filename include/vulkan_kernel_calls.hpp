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
    } // namespace detail


    class ComputeKernelRuntime
    {
      public:
        ComputeKernelRuntime(VulkanMemorySpace& space, std::string_view kernel_name, std::filesystem::path spirv_path,
            size_t num_buffer_args, size_t num_constant_args);

        ~ComputeKernelRuntime()
        {
            VkDevice dev = m_space.device();
            if (m_command_buffer != VK_NULL_HANDLE)
            {
                vkFreeCommandBuffers(dev, m_space.command_pool(), 1, &m_command_buffer);
                m_command_buffer = VK_NULL_HANDLE;
            }
            if (m_descriptor_pool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(dev, m_descriptor_pool, nullptr);
                m_descriptor_pool = VK_NULL_HANDLE;
            }
            if (m_pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(dev, m_pipeline, nullptr);
                m_pipeline = VK_NULL_HANDLE;
            }
            if (m_pipeline_layout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(dev, m_pipeline_layout, nullptr);
                m_pipeline_layout = VK_NULL_HANDLE;
            }
            if (m_descriptor_set_layout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(dev, m_descriptor_set_layout, nullptr);
                m_descriptor_set_layout = VK_NULL_HANDLE;
            }
            if (m_shader_module != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(dev, m_shader_module, nullptr);
                m_shader_module = VK_NULL_HANDLE;
            }
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
        // Cached Vulkan objects for this kernel runtime
        VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
        VkPipeline m_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
        VkShaderModule m_shader_module = VK_NULL_HANDLE;
        VkDescriptorSet m_descriptor_set = VK_NULL_HANDLE;
        VkCommandBuffer m_command_buffer = VK_NULL_HANDLE;

        int32_t m_binding_count = 1; // std::max<uint32_t>(1u, m_parsed_ssbo_binding_count);
        std::vector<VkDescriptorSetLayoutBinding> m_bindings; //(binding_count);
        std::vector<VkDescriptorBufferInfo> m_buffer_infos; //(binding_count);
        std::vector<VkWriteDescriptorSet> m_writes; //(binding_count);
        std::vector<uint32_t> m_scalar_data;
    };

    template <typename... KernelArgs>
    class ComputeKernel : public ComputeKernelRuntime
    {
      public:
        ComputeKernel(VulkanMemorySpace& space, std::string_view kernel_name, std::filesystem::path spirv_path,
            size_t num_buffer_args, size_t num_constant_args)
        : ComputeKernelRuntime(space, kernel_name, spirv_path, get_num_buffer_args(), get_num_constant_args())
        {            
        }

        // Determine scalar (int/float) arguments that should be passed via push constants.
        template <typename T>
        struct is_scalar_arg : std::bool_constant<std::is_same_v<std::decay_t<T>, int> || 
                                                  std::is_same_v<std::decay_t<T>, float> >
        {};

        static constexpr size_t get_num_buffer_args()
        {
            size_t idx = 0;
            int buffer_info_loop[] = {([&] {
                if constexpr (!is_scalar_arg<KernelArgs>::value)
                {
                    idx++;
                }
            }(), 0)...};
            (void)buffer_info_loop;
            return idx;
        }

        static constexpr size_t get_num_constant_args()
        {
            size_t scalar_count = 0;
            int scalar_count_loop[] = {((scalar_count += static_cast<size_t>(is_scalar_arg<KernelArgs>::value)), 0)...};
            (void)scalar_count_loop;
            return scalar_count;
        }


        template <typename Range>
        void launch_1d_named(Range&& range, std::initializer_list<std::string_view> parameter_names, KernelArgs... args)
        {
            // determine dispatch dimensions
            auto dims = []<typename R>(const R& r) {
                return std::tuple<size_t, size_t, size_t>{static_cast<size_t>(r.end() - r.begin()), 1u, 1u};
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
            if constexpr (requires(const Range2D& r) {
                              r.inner_size();
                              r.outer_size();
                          })
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
            if constexpr (requires(const Range3D& r) {
                              r.inner_size();
                              r.middle_size();
                              r.outer_size();
                          })
            {
                ix = range.inner_size();
                iy = range.middle_size();
                iz = range.outer_size();
            }
            else if constexpr (requires(const Range3D& r) {
                                   r.inner_size();
                                   r.outer_size();
                               })
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
            // create shader module (cached)
            VkResult r = VK_SUCCESS;
            if (m_shader_module == VK_NULL_HANDLE)
            {
                VkShaderModuleCreateInfo smci{};
                smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                smci.codeSize = m_spirv_words.size() * sizeof(uint32_t);
                smci.pCode = m_spirv_words.data();
                r = vkCreateShaderModule(m_space.device(), &smci, nullptr, &m_shader_module);
                if (r != VK_SUCCESS)
                {
                    LOG_ERROR("vkCreateShaderModule failed: {}", static_cast<int>(r));
                    std::abort();
                }
            }

            // descriptor layout
            VkDescriptorSetLayout descriptor_set_layout = m_descriptor_set_layout;
            if (descriptor_set_layout == VK_NULL_HANDLE)
            {
                VkDescriptorSetLayoutCreateInfo dslci{};
                dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                dslci.bindingCount = static_cast<uint32_t>(m_bindings.size());
                dslci.pBindings = m_bindings.data();
                r = vkCreateDescriptorSetLayout(m_space.device(), &dslci, nullptr, &descriptor_set_layout);
                if (r != VK_SUCCESS)
                {
                    LOG_ERROR("vkCreateDescriptorSetLayout failed: {}", static_cast<int>(r));
                    std::abort();
                }
                m_descriptor_set_layout = descriptor_set_layout;
            }

            constexpr size_t scalar_count = get_num_constant_args();

            VkPushConstantRange push_range{};
            if (scalar_count > 0)
            {
                push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                push_range.offset = 0;
                push_range.size = static_cast<uint32_t>(scalar_count * sizeof(uint32_t));
            }

            VkPipelineLayout pipeline_layout = m_pipeline_layout;
            if (pipeline_layout == VK_NULL_HANDLE)
            {
                VkPipelineLayoutCreateInfo plci{};
                plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                plci.setLayoutCount = 1;
                plci.pSetLayouts = &descriptor_set_layout;
                if (scalar_count > 0)
                {
                    plci.pushConstantRangeCount = 1;
                    plci.pPushConstantRanges = &push_range;
                }
                else
                {
                    plci.pushConstantRangeCount = 0;
                    plci.pPushConstantRanges = nullptr;
                }

                r = vkCreatePipelineLayout(m_space.device(), &plci, nullptr, &pipeline_layout);
                if (r != VK_SUCCESS)
                {
                    LOG_ERROR("vkCreatePipelineLayout failed: {}", static_cast<int>(r));
                    std::abort();
                }
                m_pipeline_layout = pipeline_layout;
            }

            if (m_pipeline == VK_NULL_HANDLE)
            {
                VkComputePipelineCreateInfo cpci{};
                cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                VkPipelineShaderStageCreateInfo pssci{};
                pssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                pssci.module = m_shader_module;
                pssci.pName = "main";
                cpci.stage = pssci;
                cpci.layout = pipeline_layout;

                r = vkCreateComputePipelines(m_space.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline);
                if (r != VK_SUCCESS)
                {
                    LOG_ERROR("vkCreateComputePipelines failed: {}", static_cast<int>(r));
                    std::abort();
                }
            }

            const uint32_t binding_count = static_cast<uint32_t>(m_bindings.size());

            // descriptor pool + set
            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pool_size.descriptorCount = binding_count;

            VkDescriptorPool descriptor_pool = m_descriptor_pool;
            if (descriptor_pool == VK_NULL_HANDLE)
            {
                VkDescriptorPoolCreateInfo dpci{};
                dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                dpci.maxSets = 1;
                dpci.poolSizeCount = 1;
                dpci.pPoolSizes = &pool_size;
                r = vkCreateDescriptorPool(m_space.device(), &dpci, nullptr, &descriptor_pool);
                if (r != VK_SUCCESS)
                {
                    LOG_ERROR("vkCreateDescriptorPool failed: {}", static_cast<int>(r));
                    std::abort();
                }
                m_descriptor_pool = descriptor_pool;
            }

            VkDescriptorSet descriptor_set = m_descriptor_set;
            if (descriptor_set == VK_NULL_HANDLE)
            {
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = descriptor_pool;
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts = &descriptor_set_layout;
                r = vkAllocateDescriptorSets(m_space.device(), &dsai, &descriptor_set);
                if (r != VK_SUCCESS)
                {
                    LOG_ERROR("vkAllocateDescriptorSets failed: {}", static_cast<int>(r));
                    std::abort();
                }
                m_descriptor_set = descriptor_set;
            }

            // prepare buffer infos from args (map by order). Skip scalar args.
            {
                size_t idx = 0;
                int buffer_info_loop[] = {([&] {
                    if constexpr (!is_scalar_arg<Args>::value)
                    {
                        if (idx < m_buffer_infos.size())
                            m_buffer_infos[idx++].buffer = buffer_from_arg(args);
                    }
                }(), 0)...};
                (void)buffer_info_loop;
            }
            for (auto& bi : m_buffer_infos)
            {
                bi.offset = 0;
                bi.range = VK_WHOLE_SIZE;
            }

            for (uint32_t i = 0; i < binding_count; ++i)
            {
                m_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                m_writes[i].dstSet = descriptor_set;
                m_writes[i].dstBinding = i;
                m_writes[i].dstArrayElement = 0;
                m_writes[i].descriptorCount = 1;
                m_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                m_writes[i].pBufferInfo = &m_buffer_infos[i];
            }
            vkUpdateDescriptorSets(m_space.device(), static_cast<uint32_t>(m_writes.size()), m_writes.data(), 0, nullptr);

            // Pack scalar args (int/float) into a uint32_t vector and push as push-constants
            if (scalar_count > 0)
            {
                m_scalar_data.resize(scalar_count);
                size_t sidx = 0;
                int scalar_loop[] = {([&] {
                    if constexpr (is_scalar_arg<Args>::value)
                    {
                        using Dec = std::decay_t<Args>;
                        if constexpr (std::is_same_v<Dec, int>)
                        {
                            m_scalar_data[sidx++] = static_cast<uint32_t>(args);
                        }
                        else if constexpr (std::is_same_v<Dec, float>)
                        {
                            uint32_t tmp = 0;
                            std::memcpy(&tmp, &args, sizeof(uint32_t));
                            m_scalar_data[sidx++] = tmp;
                        }
                    }
                }(), 0)...};
                (void)scalar_loop;
            }

            // allocate the cached command buffer if needed
            if (m_command_buffer == VK_NULL_HANDLE)
            {
                VkCommandBufferAllocateInfo allocate_info{};
                allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocate_info.commandPool = m_space.command_pool();
                allocate_info.commandBufferCount = 1;

                VkResult alloc_r = vkAllocateCommandBuffers(m_space.device(), &allocate_info, &m_command_buffer);
                if (alloc_r != VK_SUCCESS)
                {
                    LOG_ERROR("vkAllocateCommandBuffers failed: {}", static_cast<int>(alloc_r));
                    std::abort();
                }
            }

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            VkResult begin_r = vkBeginCommandBuffer(m_command_buffer, &begin_info);
            if (begin_r != VK_SUCCESS)
            {
                LOG_ERROR("vkBeginCommandBuffer failed: {}", static_cast<int>(begin_r));
                std::abort();
            }

            vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
            vkCmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
            if (!m_scalar_data.empty())
            {
                vkCmdPushConstants(m_command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(m_scalar_data.size() * sizeof(uint32_t)), m_scalar_data.data());
            }
            vkCmdDispatch(m_command_buffer, gx, gy, gz);

            VkResult end_r = vkEndCommandBuffer(m_command_buffer);
            if (end_r != VK_SUCCESS)
            {
                LOG_ERROR("vkEndCommandBuffer failed: {}", static_cast<int>(end_r));
                std::abort();
            }

            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &m_command_buffer;

            VkResult submit_r = vkQueueSubmit(m_space.queue(), 1, &submit_info, VK_NULL_HANDLE);
            if (submit_r != VK_SUCCESS)
            {
                LOG_ERROR("vkQueueSubmit failed: {}", static_cast<int>(submit_r));
                std::abort();
            }

            VkResult wait_r = vkQueueWaitIdle(m_space.queue());
            if (wait_r != VK_SUCCESS)
            {
                LOG_ERROR("vkQueueWaitIdle failed: {}", static_cast<int>(wait_r));
                std::abort();
            }

            // cached Vulkan objects are reused; cleanup is handled in the runtime destructor
        }
    };
} // namespace rllm::vulkan

#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <cstdlib>
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
#include <type_traits>
#include <utility>
#include <vector>

#include <device_pointer.hpp>
#include <logging.hpp>
#include <parallel.hpp>
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
            const void* offload_ptr = nullptr;
            size_t size_bytes = 0;
            std::string_view parameter_name{};
            std::function<DeviceMemoryOwner()> memory_owner;
            std::function<void()> mark_device_latest;
            // Called with (mapped_ptr, size_bytes, kernel_name) once the Vulkan device buffer is ready.
            // Allows the caller to register deferred host synchronization or eager copy-back.
            std::function<void(void*, size_t, std::string_view, std::string_view)> on_device_ready;
#if defined(USE_VULKAN_OFFLOAD)
            VulkanRuntimeBuffer* vulkan_runtime_buffer = nullptr;
#endif
        };

        template <typename T>
        struct is_device_pointer : std::false_type {};
        template <typename T>
        struct is_device_pointer<DevicePointer<T>> : std::true_type {};
        template <typename T>
        inline constexpr bool is_device_pointer_v = is_device_pointer<std::remove_cv_t<std::remove_reference_t<T>>>::value;

        template <typename T>
        inline constexpr bool is_lazy_host_buffer_arg_v = !is_device_pointer_v<T> && requires(T& value, std::function<void()> flush_fn) {
            value.raw_staging_data();
            value.storage_size_bytes();
            value.set_pending_flush(std::move(flush_fn));
            value.needs_offload_sync();
        };

        template <typename T>
        inline constexpr bool is_offload_backed_buffer_arg_v = !is_device_pointer_v<T> && requires(T& value) {
            value.raw_staging_data();
            value.raw_offload_data();
            value.device_memory_owner();
            value.storage_size_bytes();
            value.needs_offload_sync();
        };

        template <typename T>
        inline constexpr bool is_host_buffer_arg_v = !is_device_pointer_v<T> && !is_offload_backed_buffer_arg_v<T> && requires(T& value) {
            value.data();
            value.storage_size_bytes();
        };

        template <typename T>
        inline constexpr bool is_contiguous_container_arg_v = !is_device_pointer_v<T> && !is_host_buffer_arg_v<T> && requires(T& value) {
            value.data();
            value.size();
        };

        template <typename T>
        inline constexpr bool is_scalar_kernel_arg_v =
            !is_device_pointer_v<T> && !is_host_buffer_arg_v<T> && !is_contiguous_container_arg_v<T> &&
            (std::is_enum_v<std::remove_cv_t<std::remove_reference_t<T>>> ||
             std::is_integral_v<std::remove_cv_t<std::remove_reference_t<T>>> ||
             std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<T>>>);

        template <size_t N, typename Arg>
        inline void append_scalar_arg(std::array<std::byte, N>& out, size_t& out_size, Arg&& arg)
        {
            using ArgType = std::remove_cv_t<std::remove_reference_t<Arg>>;
            if constexpr (is_scalar_kernel_arg_v<ArgType>)
            {
                if (out_size + sizeof(uint32_t) > N)
                {
                    LOG_ERROR("Vulkan launch scalar buffer overflow: capacity={}, attempted to append one more scalar argument.", N);
                    std::abort();
                }

                if constexpr (std::is_floating_point_v<ArgType>)
                {
                    const float value = static_cast<float>(arg);
                    std::memcpy(out.data() + out_size, &value, sizeof(value));
                }
                else if constexpr (std::is_enum_v<ArgType>)
                {
                    const int32_t value = static_cast<int32_t>(arg);
                    std::memcpy(out.data() + out_size, &value, sizeof(value));
                }
                else
                {
                    const int32_t value = static_cast<int32_t>(arg);
                    std::memcpy(out.data() + out_size, &value, sizeof(value));
                }

                out_size += sizeof(uint32_t);
            }
            else
            {
                static_cast<void>(out);
                static_cast<void>(out_size);
                static_cast<void>(arg);
            }
        }

        template <typename Arg>
        inline void append_buffer_view(std::vector<HostBufferView>& out, std::string_view parameter_name, Arg&& arg)
        {
            using ArgType = std::remove_reference_t<Arg>;
            if constexpr (is_device_pointer_v<ArgType>)
            {
                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(arg.raw_staging_data()));
                view.offload_ptr = arg.raw_offload_data();
                view.size_bytes = arg.storage_size_bytes();
                view.parameter_name = parameter_name;
                view.memory_owner = [&arg]() { return arg.device_memory_owner(); };
                view.mark_device_latest = [&arg]() {
                    const_cast<std::remove_const_t<ArgType>&>(arg).mark_device_latest();
                };
#if defined(USE_VULKAN_OFFLOAD)
                view.vulkan_runtime_buffer = &const_cast<std::remove_const_t<ArgType>&>(arg).vulkan_runtime_buffer();
#endif
                if constexpr (!std::is_const_v<ArgType>)
                {
                    auto* arg_ptr = std::addressof(arg);
                    view.on_device_ready = [arg_ptr](
                        void* mapped_ptr,
                        size_t sz,
                        std::string_view kernel_name,
                        std::string_view parameter_name_value
                    ) {
                        arg_ptr->set_pending_flush([arg_ptr, mapped_ptr, sz, kernel_name, parameter_name_value]() {
                            parallel::statistics.record_device_to_host_buffer_copy(kernel_name, parameter_name_value, sz);
                            std::memcpy(arg_ptr->raw_staging_data(), mapped_ptr, sz);
                        });
                    };
                }
                out.push_back(view);
            }
            else if constexpr (is_offload_backed_buffer_arg_v<ArgType>)
            {
                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(arg.raw_staging_data()));
                view.offload_ptr = arg.raw_offload_data();
                view.size_bytes = static_cast<size_t>(arg.storage_size_bytes());
                view.parameter_name = parameter_name;
                view.memory_owner = [&arg]() { return arg.device_memory_owner(); };
                view.mark_device_latest = [&arg]() {
                    const_cast<std::remove_const_t<ArgType>&>(arg).mark_device_latest();
                };
#if defined(USE_VULKAN_OFFLOAD)
                if constexpr (requires { const_cast<std::remove_const_t<ArgType>&>(arg).vulkan_runtime_buffer(); })
                    view.vulkan_runtime_buffer = &const_cast<std::remove_const_t<ArgType>&>(arg).vulkan_runtime_buffer();
#endif
                if constexpr (!std::is_const_v<ArgType> && is_lazy_host_buffer_arg_v<ArgType>)
                {
                    auto* arg_ptr = std::addressof(arg);
                    view.on_device_ready = [arg_ptr](
                        void* mapped_ptr,
                        size_t sz,
                        std::string_view kernel_name,
                        std::string_view parameter_name_value
                    ) {
                        arg_ptr->set_pending_flush([arg_ptr, mapped_ptr, sz, kernel_name, parameter_name_value]() {
                            parallel::statistics.record_device_to_host_buffer_copy(kernel_name, parameter_name_value, sz);
                            std::memcpy(arg_ptr->raw_staging_data(), mapped_ptr, sz);
                        });
                    };
                }
                out.push_back(view);
            }
            else if constexpr (is_host_buffer_arg_v<ArgType>)
            {
                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(arg.data()));
                view.size_bytes = static_cast<size_t>(arg.storage_size_bytes());
                view.parameter_name = parameter_name;
                if constexpr (!std::is_const_v<ArgType>)
                {
                    view.on_device_ready = [&arg](
                        void* mapped_ptr,
                        size_t sz,
                        std::string_view kernel_name,
                        std::string_view parameter_name_value
                    ) {
                        parallel::statistics.record_device_to_host_buffer_copy(kernel_name, parameter_name_value, sz);
                        std::memcpy(arg.data(), mapped_ptr, sz);
                    };
                }
                out.push_back(view);
            }
            else if constexpr (is_contiguous_container_arg_v<ArgType>)
            {
                auto* ptr = arg.data();
                using ElementType = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;

                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(ptr));
                view.size_bytes = static_cast<size_t>(arg.size()) * sizeof(ElementType);
                view.parameter_name = parameter_name;
                if constexpr (!std::is_const_v<std::remove_pointer_t<decltype(ptr)>>)
                {
                    view.on_device_ready = [ptr](
                        void* mapped_ptr,
                        size_t sz,
                        std::string_view kernel_name,
                        std::string_view parameter_name_value
                    ) {
                        parallel::statistics.record_device_to_host_buffer_copy(kernel_name, parameter_name_value, sz);
                        std::memcpy(ptr, mapped_ptr, sz);
                    };
                }
                out.push_back(view);
            }
            else
            {
                static_cast<void>(out);
                static_cast<void>(parameter_name);
                static_cast<void>(arg);
            }
        }

        template <size_t N, typename Arg>
        inline void append_buffer_view(
            std::array<HostBufferView, N>& out,
            size_t& out_count,
            std::string_view parameter_name,
            Arg&& arg
        )
        {
            using ArgType = std::remove_reference_t<Arg>;
            if constexpr (is_device_pointer_v<ArgType>)
            {
                if (out_count >= N)
                {
                    LOG_ERROR(
                        "Vulkan launch argument buffer overflow: capacity={}, attempted to append one more argument.", N
                    );
                    std::abort();
                }

                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(arg.raw_staging_data()));
                view.offload_ptr = arg.raw_offload_data();
                view.size_bytes = arg.storage_size_bytes();
                view.parameter_name = parameter_name;
                view.memory_owner = [&arg]() { return arg.device_memory_owner(); };
                view.mark_device_latest = [&arg]() {
                    const_cast<std::remove_const_t<ArgType>&>(arg).mark_device_latest();
                };
#if defined(USE_VULKAN_OFFLOAD)
                view.vulkan_runtime_buffer = &const_cast<std::remove_const_t<ArgType>&>(arg).vulkan_runtime_buffer();
#endif
                if constexpr (!std::is_const_v<ArgType>)
                {
                    auto* arg_ptr = std::addressof(arg);
                    view.on_device_ready = [arg_ptr](
                        void* mapped_ptr,
                        size_t sz,
                        std::string_view kernel_name,
                        std::string_view parameter_name_value
                    ) {
                        arg_ptr->set_pending_flush([arg_ptr, mapped_ptr, sz, kernel_name, parameter_name_value]() {
                            parallel::statistics.record_device_to_host_buffer_copy(kernel_name, parameter_name_value, sz);
                            std::memcpy(arg_ptr->raw_staging_data(), mapped_ptr, sz);
                        });
                    };
                }
                out[out_count++] = view;
            }
            else if constexpr (is_offload_backed_buffer_arg_v<ArgType>)
            {
                if (out_count >= N)
                {
                    LOG_ERROR(
                        "Vulkan launch argument buffer overflow: capacity={}, attempted to append one more argument.", N
                    );
                    std::abort();
                }

                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(arg.raw_staging_data()));
                view.offload_ptr = arg.raw_offload_data();
                view.size_bytes = static_cast<size_t>(arg.storage_size_bytes());
                view.parameter_name = parameter_name;
                view.memory_owner = [&arg]() { return arg.device_memory_owner(); };
                view.mark_device_latest = [&arg]() {
                    const_cast<std::remove_const_t<ArgType>&>(arg).mark_device_latest();
                };
#if defined(USE_VULKAN_OFFLOAD)
                if constexpr (requires { const_cast<std::remove_const_t<ArgType>&>(arg).vulkan_runtime_buffer(); })
                    view.vulkan_runtime_buffer = &const_cast<std::remove_const_t<ArgType>&>(arg).vulkan_runtime_buffer();
#endif
                if constexpr (!std::is_const_v<ArgType> && is_lazy_host_buffer_arg_v<ArgType>)
                {
                    auto* arg_ptr = std::addressof(arg);
                    view.on_device_ready = [arg_ptr](
                        void* mapped_ptr,
                        size_t sz,
                        std::string_view kernel_name,
                        std::string_view parameter_name_value
                    ) {
                        arg_ptr->set_pending_flush([arg_ptr, mapped_ptr, sz, kernel_name, parameter_name_value]() {
                            parallel::statistics.record_device_to_host_buffer_copy(kernel_name, parameter_name_value, sz);
                            std::memcpy(arg_ptr->raw_staging_data(), mapped_ptr, sz);
                        });
                    };
                }
                out[out_count++] = view;
            }
            else if constexpr (is_host_buffer_arg_v<ArgType>)
            {
                if (out_count >= N)
                {
                    LOG_ERROR(
                        "Vulkan launch argument buffer overflow: capacity={}, attempted to append one more argument.", N
                    );
                    std::abort();
                }

                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(arg.data()));
                view.size_bytes = static_cast<size_t>(arg.storage_size_bytes());
                view.parameter_name = parameter_name;
                if constexpr (!std::is_const_v<ArgType>)
                {
                    view.on_device_ready = [&arg](
                        void* mapped_ptr,
                        size_t sz,
                        std::string_view kernel_name,
                        std::string_view parameter_name_value
                    ) {
                        parallel::statistics.record_device_to_host_buffer_copy(kernel_name, parameter_name_value, sz);
                        std::memcpy(arg.data(), mapped_ptr, sz);
                    };
                }
                out[out_count++] = view;
            }
            else if constexpr (is_contiguous_container_arg_v<ArgType>)
            {
                if (out_count >= N)
                {
                    LOG_ERROR(
                        "Vulkan launch argument buffer overflow: capacity={}, attempted to append one more argument.", N
                    );
                    std::abort();
                }

                auto* ptr = arg.data();
                using ElementType = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;

                HostBufferView view{};
                view.host_ptr = const_cast<void*>(static_cast<const void*>(ptr));
                view.size_bytes = static_cast<size_t>(arg.size()) * sizeof(ElementType);
                view.parameter_name = parameter_name;
                if constexpr (!std::is_const_v<std::remove_pointer_t<decltype(ptr)>>)
                {
                    view.on_device_ready = [ptr](
                        void* mapped_ptr,
                        size_t sz,
                        std::string_view kernel_name,
                        std::string_view parameter_name_value
                    ) {
                        parallel::statistics.record_device_to_host_buffer_copy(kernel_name, parameter_name_value, sz);
                        std::memcpy(ptr, mapped_ptr, sz);
                    };
                }
                out[out_count++] = view;
            }
            else
            {
                static_cast<void>(out);
                static_cast<void>(out_count);
                static_cast<void>(parameter_name);
                static_cast<void>(arg);
            }
        }

        std::vector<uint32_t> load_spirv_words(const std::filesystem::path& spirv, const char* kernel_name);

        struct LocalSize
        {
            uint32_t x = 1;
            uint32_t y = 1;
            uint32_t z = 1;
        };

        uint32_t count_ssbo_bindings_in_glsl(const std::filesystem::path& spirv_path);
        LocalSize parse_local_size_from_glsl(const std::filesystem::path& spirv_path);
    } // namespace detail

    class ComputeKernelRuntime
    {
      protected:
        struct RuntimeBuffer;

      public:
        ComputeKernelRuntime(std::string_view kernel_name, std::filesystem::path spirv_path);

        ~ComputeKernelRuntime();

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

      protected:
        void dispatch_kernel(
            uint32_t groups_x,
            uint32_t groups_y,
            std::span<const detail::HostBufferView> buffers,
            std::span<const std::byte> push_constants,
            std::span<RuntimeBuffer> runtime_buffers,
            size_t& runtime_buffer_count
        );
        void ensure_pipeline(VkDevice device, uint32_t ssbo_binding_count);
        void wait_for_in_flight_submit();

        struct RuntimeBuffer
        {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VkDeviceMemory staging_memory = VK_NULL_HANDLE;
            void* mapped = nullptr;
            detail::HostBufferView view{};
            bool cached = false;  // true when buffer is kept alive for lazy-flush DevicePointer args
            bool use_offload_source = false;
            bool bind_offload_direct = false;
            VkDeviceSize offload_source_offset = 0;
        };

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
        bool m_submit_in_flight = false;
        std::mutex m_launch_mutex;
    };

    template <typename... KernelArgs>
    class ComputeKernel : public ComputeKernelRuntime
    {
      public:
        using ComputeKernelRuntime::ComputeKernelRuntime;

        template <typename Range>
        void launch_1d(Range&& range, KernelArgs... args);

                template <typename Range>
                void launch_1d_named(Range&& range, std::initializer_list<std::string_view> parameter_names, KernelArgs... args);

        template <typename Range2D>
        void launch_2d(Range2D&& range, KernelArgs... args);

                template <typename Range2D>
                void launch_2d_named(
                        Range2D&& range,
                        std::initializer_list<std::string_view> parameter_names,
                        KernelArgs... args
                );

      private:
                template <size_t... I>
                void append_named_buffer_views(
                        std::index_sequence<I...>,
                        const std::array<std::string_view, sizeof...(KernelArgs)>& parameter_names,
                        KernelArgs... args
                )
                {
                        (detail::append_buffer_view(m_buffers, m_buffer_count, parameter_names[I], args), ...);
                }

        static constexpr size_t kLaunchBufferCount = sizeof...(KernelArgs);
                static constexpr size_t kLaunchScalarBytes = (sizeof...(KernelArgs) + 2) * sizeof(uint32_t);
        std::array<detail::HostBufferView, kLaunchBufferCount> m_buffers{};
        size_t m_buffer_count = 0;
                std::array<std::byte, kLaunchScalarBytes> m_push_constant_bytes{};
                size_t m_push_constant_size = 0;
        std::array<typename ComputeKernelRuntime::RuntimeBuffer, kLaunchBufferCount> m_runtime_buffers{};
        size_t m_runtime_buffer_count = 0;
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
            VkBuffer offload_buffer = VK_NULL_HANDLE;
            const void* offload_base = nullptr;
            size_t offload_size = 0;
            VkDeviceSize storage_buffer_offset_alignment = 1;
            std::shared_ptr<std::mutex> launch_mutex = std::make_shared<std::mutex>();
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
            else if constexpr (requires {
                                   range.hi;
                                   range.lo;
                               })
            {
                count = to_size(range.hi - range.lo);
            }
            else if constexpr (requires {
                                   range.end();
                                   range.begin();
                               })
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

            if constexpr (requires {
                              range.inner_size();
                              range.outer_size();
                          })
            {
                x = to_size(range.outer_size());
                y = to_size(range.inner_size());
            }
            else if constexpr (requires {
                                   range.cols();
                                   range.rows();
                               })
            {
                x = to_size(range.rows());
                y = to_size(range.cols());
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

    template <typename... KernelArgs>
    template <typename Range>
    inline void ComputeKernel<KernelArgs...>::launch_1d(Range&& range, KernelArgs... args)
    {
        std::lock_guard<std::mutex> lock(m_launch_mutex);

        if (!std::filesystem::exists(spirv_path()))
        {
            LOG_ERROR("Missing generated Vulkan kernel artifact '{}' for kernel '{}'.", spirv_path().string(), name());
            std::abort();
        }

        const uint32_t x_items = detail::range_size_1d(range);
        const uint32_t groups_x = detail::ceil_div_u32(x_items, local_size_x());
        const std::array<std::string_view, sizeof...(KernelArgs)> parameter_names{};

        m_buffer_count = 0;
        m_push_constant_size = 0;
        append_named_buffer_views(std::index_sequence_for<KernelArgs...>{}, parameter_names, args...);
        detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, static_cast<int32_t>(x_items));
        detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, static_cast<int32_t>(1));
        (detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, args), ...);

        dispatch_kernel(
            groups_x,
            1,
            std::span<const detail::HostBufferView>(m_buffers.data(), m_buffer_count),
            std::span<const std::byte>(m_push_constant_bytes.data(), m_push_constant_size),
            std::span<typename ComputeKernelRuntime::RuntimeBuffer>(m_runtime_buffers.data(), m_runtime_buffers.size()),
            m_runtime_buffer_count
        );
    }

    template <typename... KernelArgs>
    template <typename Range2D>
    inline void ComputeKernel<KernelArgs...>::launch_2d(Range2D&& range, KernelArgs... args)
    {
        std::lock_guard<std::mutex> lock(m_launch_mutex);

        if (!std::filesystem::exists(spirv_path()))
        {
            LOG_ERROR("Missing generated Vulkan kernel artifact '{}' for kernel '{}'.", spirv_path().string(), name());
            std::abort();
        }

        const auto [x_items, y_items] = detail::range_size_2d(range);
        const uint32_t groups_x = detail::ceil_div_u32(x_items, local_size_x());
        const uint32_t groups_y = detail::ceil_div_u32(y_items, local_size_y());
        const std::array<std::string_view, sizeof...(KernelArgs)> parameter_names{};

        m_buffer_count = 0;
        m_push_constant_size = 0;
        append_named_buffer_views(std::index_sequence_for<KernelArgs...>{}, parameter_names, args...);
        detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, static_cast<int32_t>(x_items));
        detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, static_cast<int32_t>(y_items));
        (detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, args), ...);

        dispatch_kernel(
            groups_x,
            groups_y,
            std::span<const detail::HostBufferView>(m_buffers.data(), m_buffer_count),
            std::span<const std::byte>(m_push_constant_bytes.data(), m_push_constant_size),
            std::span<typename ComputeKernelRuntime::RuntimeBuffer>(m_runtime_buffers.data(), m_runtime_buffers.size()),
            m_runtime_buffer_count
        );
    }

    template <typename... KernelArgs>
    template <typename Range>
    inline void ComputeKernel<KernelArgs...>::launch_1d_named(
        Range&& range,
        std::initializer_list<std::string_view> parameter_names_list,
        KernelArgs... args
    )
    {
        std::lock_guard<std::mutex> lock(m_launch_mutex);

        if (!std::filesystem::exists(spirv_path()))
        {
            LOG_ERROR("Missing generated Vulkan kernel artifact '{}' for kernel '{}'.", spirv_path().string(), name());
            std::abort();
        }

        if (parameter_names_list.size() != sizeof...(KernelArgs))
        {
            LOG_ERROR(
                "Kernel '{}' expected {} parameter names, but launch provided {}.",
                name(),
                sizeof...(KernelArgs),
                parameter_names_list.size()
            );
            std::abort();
        }

        std::array<std::string_view, sizeof...(KernelArgs)> parameter_names{};
        std::copy(parameter_names_list.begin(), parameter_names_list.end(), parameter_names.begin());

        const uint32_t x_items = detail::range_size_1d(range);
        const uint32_t groups_x = detail::ceil_div_u32(x_items, local_size_x());

        m_buffer_count = 0;
        m_push_constant_size = 0;
        append_named_buffer_views(std::index_sequence_for<KernelArgs...>{}, parameter_names, args...);
        detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, static_cast<int32_t>(x_items));
        detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, static_cast<int32_t>(1));
        (detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, args), ...);

        dispatch_kernel(
            groups_x,
            1,
            std::span<const detail::HostBufferView>(m_buffers.data(), m_buffer_count),
            std::span<const std::byte>(m_push_constant_bytes.data(), m_push_constant_size),
            std::span<typename ComputeKernelRuntime::RuntimeBuffer>(m_runtime_buffers.data(), m_runtime_buffers.size()),
            m_runtime_buffer_count
        );
    }

    template <typename... KernelArgs>
    template <typename Range2D>
    inline void ComputeKernel<KernelArgs...>::launch_2d_named(
        Range2D&& range,
        std::initializer_list<std::string_view> parameter_names_list,
        KernelArgs... args
    )
    {
        std::lock_guard<std::mutex> lock(m_launch_mutex);

        if (!std::filesystem::exists(spirv_path()))
        {
            LOG_ERROR("Missing generated Vulkan kernel artifact '{}' for kernel '{}'.", spirv_path().string(), name());
            std::abort();
        }

        if (parameter_names_list.size() != sizeof...(KernelArgs))
        {
            LOG_ERROR(
                "Kernel '{}' expected {} parameter names, but launch provided {}.",
                name(),
                sizeof...(KernelArgs),
                parameter_names_list.size()
            );
            std::abort();
        }

        std::array<std::string_view, sizeof...(KernelArgs)> parameter_names{};
        std::copy(parameter_names_list.begin(), parameter_names_list.end(), parameter_names.begin());

        const auto [x_items, y_items] = detail::range_size_2d(range);
        const uint32_t groups_x = detail::ceil_div_u32(x_items, local_size_x());
        const uint32_t groups_y = detail::ceil_div_u32(y_items, local_size_y());

        m_buffer_count = 0;
        m_push_constant_size = 0;
        append_named_buffer_views(std::index_sequence_for<KernelArgs...>{}, parameter_names, args...);
        detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, static_cast<int32_t>(x_items));
        detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, static_cast<int32_t>(y_items));
        (detail::append_scalar_arg(m_push_constant_bytes, m_push_constant_size, args), ...);

        dispatch_kernel(
            groups_x,
            groups_y,
            std::span<const detail::HostBufferView>(m_buffers.data(), m_buffer_count),
            std::span<const std::byte>(m_push_constant_bytes.data(), m_push_constant_size),
            std::span<typename ComputeKernelRuntime::RuntimeBuffer>(m_runtime_buffers.data(), m_runtime_buffers.size()),
            m_runtime_buffer_count
        );
    }

} // namespace rllm::vulkan

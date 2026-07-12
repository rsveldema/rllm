#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>

#include <rllm_vulkan_runtime.hpp>

namespace rllm
{
    /** Plain host-only data storage backed by a Vulkan HOST_VISIBLE|HOST_COHERENT
     *  buffer.  Because the buffer is already pinned in a Vulkan-accessible region
     *  the gpu_* copy_from_cpu / copy_to_cpu helpers can transfer directly from/to
     *  this buffer without going through a second staging allocation.
     */
    template <typename T>
    class cpu_data
    {
    public:
        explicit cpu_data(size_t elts)
            : m_size(elts)
            , m_host(rllm::vulkan_runtime::session(), static_cast<VkDeviceSize>(sizeof(T) * elts))
        {}

        cpu_data(const cpu_data& other)
            : m_size(other.m_size)
            , m_host(rllm::vulkan_runtime::session(), static_cast<VkDeviceSize>(sizeof(T) * other.m_size))
        {
            std::memcpy(data(), other.data(), m_size * sizeof(T));
        }

        cpu_data& operator=(const cpu_data& other)
        {
            if (this != &other)
                std::memcpy(data(), other.data(), std::min(m_size, other.m_size) * sizeof(T));
            return *this;
        }

        cpu_data(cpu_data&&) noexcept = default;
        cpu_data& operator=(cpu_data&&) noexcept = default;
        ~cpu_data() = default;

        void zero()
        {
            std::fill_n(data(), m_size, T{});
        }

        T* data()             { return reinterpret_cast<T*>(m_host.bytes()); }
        const T* data() const { return reinterpret_cast<const T*>(m_host.bytes()); }

        /** Return the underlying Vulkan host buffer for direct H2D / D2H transfers. */
        VBaseHostBuffer& vk_host_buffer()             { return m_host; }
        const VBaseHostBuffer& vk_host_buffer() const { return m_host; }

        void resize(size_t new_size)
        {
            if (new_size == m_size)
                return;
            VDynamicHostBuffer new_host(rllm::vulkan_runtime::session(),
                                        static_cast<VkDeviceSize>(sizeof(T) * new_size));
            std::memcpy(new_host.bytes(), m_host.bytes(),
                        std::min(m_size, new_size) * sizeof(T));
            m_host = std::move(new_host);
            m_size = new_size;
        }

    protected:
        size_t m_size;
        VDynamicHostBuffer m_host;
    };
} // namespace rllm


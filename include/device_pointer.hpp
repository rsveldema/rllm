#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <rllm_vulkan_runtime.hpp>
#include <parallel.hpp>


template <typename T>
class DevicePointer
{
public:
    explicit DevicePointer(size_t num_elements)
        : m_count(num_elements)
        , m_bytes(sizeof(T) * num_elements)
        , m_device(std::make_unique<VDynamicDeviceBuffer>(
                       rllm::vulkan_runtime::session(),
                       static_cast<VkDeviceSize>(m_bytes)))
    {
        assert(m_count > 0);
        m_device->zero(rllm::vulkan_runtime::context());
    }

    DevicePointer(const DevicePointer&) = delete;
    DevicePointer& operator=(const DevicePointer&) = delete;

    DevicePointer(DevicePointer&& other) noexcept
    {
        std::lock_guard<std::mutex> other_lock(other.m_state_mutex);
        m_count = other.m_count;
        m_bytes = other.m_bytes;
        m_device = std::move(other.m_device);
        m_last_device_writer = std::move(other.m_last_device_writer);
        other.m_count = 0;
        other.m_bytes = 0;
    }

    DevicePointer& operator=(DevicePointer&& other) noexcept
    {
        if (this != &other)
        {
            std::lock_guard<std::mutex> other_lock(other.m_state_mutex);
            m_count = other.m_count;
            m_bytes = other.m_bytes;
            m_device = std::move(other.m_device);
            m_last_device_writer = std::move(other.m_last_device_writer);
            other.m_count = 0;
            other.m_bytes = 0;
        }
        return *this;
    }

    ~DevicePointer() = default;

    void zero()
    {
        m_device->zero(rllm::vulkan_runtime::context());
    }

    size_t size() const { return m_count; }
    size_t storage_size_bytes() const { return m_bytes; }

    /** H2D: upload from an external host-visible Vulkan buffer. */
    void copy_to_offload_buffer(VBaseHostBuffer& src,
                                std::string_view site = {},
                                std::string_view parameter = {})
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_device->write(rllm::vulkan_runtime::context(), src);
        parallel::statistics.record_host_to_device_buffer_copy(site, parameter, m_bytes);
        ComputeKernelRegistry::instance().recordHostToDevice(ComputeKernelRegistry::activeKernel(), m_bytes);
    }

    /** D2H: download to an external host-visible Vulkan buffer. */
    void copy_from_offload_buffer(VBaseHostBuffer& dst,
                                  std::string_view = {},
                                  std::string_view = {}) const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_device->read(rllm::vulkan_runtime::context(), dst);
        parallel::statistics.record_device_to_host_buffer_copy({}, {}, m_bytes);
        ComputeKernelRegistry::instance().recordDeviceToHost(m_last_device_writer, m_bytes);
    }

    /** H2D partial: upload a contiguous element range from an external host buffer. */
    void copy_range_to_offload_buffer(VBaseHostBuffer& src,
                                      size_t start_element,
                                      size_t element_count,
                                      std::string_view site = {},
                                      std::string_view parameter = {})
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        assert(start_element <= m_count);
        assert(element_count <= m_count - start_element);
        const auto offset = static_cast<VkDeviceSize>(start_element * sizeof(T));
        const auto bytes  = static_cast<VkDeviceSize>(element_count  * sizeof(T));
        m_device->write(rllm::vulkan_runtime::context(), src, bytes, offset, offset);
        parallel::statistics.record_host_to_device_buffer_copy(site, parameter, static_cast<size_t>(bytes));
        ComputeKernelRegistry::instance().recordHostToDevice(ComputeKernelRegistry::activeKernel(), static_cast<size_t>(bytes));
    }

    void mark_device_latest()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_last_device_writer = std::string(ComputeKernelRegistry::activeKernelName());
    }

    VBaseDeviceBuffer& device_buffer() const
    {
        return *m_device;
    }

private:
    size_t m_count = 0;
    size_t m_bytes = 0;
    std::unique_ptr<VDynamicDeviceBuffer> m_device;
    mutable std::mutex m_state_mutex;
    mutable std::string m_last_device_writer;
};

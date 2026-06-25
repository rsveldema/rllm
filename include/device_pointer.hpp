#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <rllm_vulkan_runtime.hpp>
#include <parallel.hpp>

enum class DeviceMemoryOwner {
    INVALID,
    ON_DEVICE,
    ON_HOST,
    REPLICATED,
};

template <typename T>
class DevicePointer
{
public:
    explicit DevicePointer(size_t num_elements)
        : m_count(num_elements)
        , m_bytes(sizeof(T) * num_elements)
    {
        assert(m_count > 0);
        allocate_internal();
        zero();
    }

    DevicePointer(const DevicePointer& other)
        : m_count(other.m_count)
        , m_bytes(other.m_bytes)
    {
        allocate_internal();
        copy_contents_from(other);
    }

    DevicePointer& operator=(const DevicePointer& other)
    {
        if (this == &other)
            return *this;
        resize(other.m_count);
        copy_contents_from(other);
        return *this;
    }

    DevicePointer(DevicePointer&& other) noexcept
    {
        move_from(std::move(other));
    }

    DevicePointer& operator=(DevicePointer&& other) noexcept
    {
        if (this != &other)
            move_from(std::move(other));
        return *this;
    }
    ~DevicePointer() = default;

    void zero()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        assert(m_bytes > 0);
        std::memset(host_bytes(), 0, m_bytes);
        m_device->write(rllm::vulkan_runtime::context(), *m_host);
        m_owner = DeviceMemoryOwner::REPLICATED;
    }

    void fill(T value)
    {
        ensure_host_data();
        std::fill_n(host_data(), m_count, value);
        mark_host_modified();
    }

    T* get() const
    {
        ensure_host_data();
        mark_host_modified();
        return host_data();
    }

    T* staging_data() const { return get(); }
    T* raw_staging_data() const
    {
        ensure_host_data();
        return host_data();
    }

    size_t size() const { return m_count; }
    size_t storage_size_bytes() const { return m_bytes; }

    void resize(size_t num_elements)
    {
        assert(num_elements > 0);
        if (num_elements == m_count)
            return;

        DevicePointer resized(num_elements);
        const size_t copy_count = std::min(m_count, num_elements);
        if (copy_count > 0)
        {
            ensure_host_data();
            std::memcpy(resized.host_bytes(), host_bytes(), copy_count * sizeof(T));
            resized.m_owner = DeviceMemoryOwner::ON_HOST;
        }
        *this = std::move(resized);
    }

    T* data() { return get(); }
    const T* data() const
    {
        ensure_host_data();
        return host_data();
    }

    void set_pending_flush(std::function<void()> flush_fn)
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_pending_flush = std::move(flush_fn);
        m_owner = DeviceMemoryOwner::ON_DEVICE;
    }

    void mark_device_latest()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_pending_flush = nullptr;
        m_owner = DeviceMemoryOwner::ON_DEVICE;
        m_last_device_writer = std::string(ComputeKernelRegistry::activeKernelName());
    }

    DeviceMemoryOwner device_memory_owner() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return m_owner;
    }

    bool has_pending_flush() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return static_cast<bool>(m_pending_flush);
    }

    T& operator[](size_t idx)
    {
        assert(idx < m_count);
        ensure_host_data();
        mark_host_modified();
        return host_data()[idx];
    }

    const T& operator[](size_t idx) const
    {
        assert(idx < m_count);
        ensure_host_data();
        return host_data()[idx];
    }

    T get(size_t idx) const
    {
        assert(idx < m_count);
        ensure_host_data();
        return host_data()[idx];
    }

    T get_offload_synced(size_t idx) const
    {
        assert(idx < m_count);
        ensure_host_data();
        return host_data()[idx];
    }

    void set(size_t idx, const T& value)
    {
        assert(idx < m_count);
        ensure_host_data();
        host_data()[idx] = value;
        mark_host_modified();
    }

    void copy_to_offload_buffer(std::string_view site = {}, std::string_view parameter = {})
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        copy_to_offload_buffer_unlocked(site, parameter);
    }

    void copy_range_to_offload_buffer(size_t start_element, size_t element_count, std::string_view site = {}, std::string_view parameter = {})
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        assert(start_element <= m_count);
        assert(element_count <= m_count - start_element);
        if (m_owner == DeviceMemoryOwner::ON_HOST || m_owner == DeviceMemoryOwner::INVALID)
        {
            const auto offset = static_cast<VkDeviceSize>(start_element * sizeof(T));
            const auto bytes = static_cast<VkDeviceSize>(element_count * sizeof(T));
            m_device->write(rllm::vulkan_runtime::context(), *m_host, bytes, offset, offset);
            parallel::statistics.record_host_to_device_buffer_copy(site, parameter, static_cast<size_t>(bytes));
            ComputeKernelRegistry::instance().recordHostToDevice(ComputeKernelRegistry::activeKernel(), static_cast<size_t>(bytes));
            m_owner = DeviceMemoryOwner::REPLICATED;
        }
    }

    void copy_from_offload_buffer(std::string_view = {}, std::string_view = {})
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        copy_from_offload_buffer_unlocked();
    }

    bool needs_offload_sync() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return m_owner == DeviceMemoryOwner::ON_HOST || m_owner == DeviceMemoryOwner::INVALID;
    }

    VBaseDeviceBuffer& device_buffer() const
    {
        ensure_device_data();
        return *m_device;
    }

private:
    void allocate_internal()
    {
        m_host = std::make_unique<VDynamicHostBuffer>(rllm::vulkan_runtime::session(), static_cast<VkDeviceSize>(m_bytes));
        m_device = std::make_unique<VDynamicDeviceBuffer>(rllm::vulkan_runtime::session(), static_cast<VkDeviceSize>(m_bytes));
        m_owner = DeviceMemoryOwner::ON_HOST;
    }

    void copy_contents_from(const DevicePointer& other)
    {
        other.ensure_host_data();
        std::memcpy(host_bytes(), other.host_bytes(), std::min(m_bytes, other.m_bytes));
        m_owner = DeviceMemoryOwner::ON_HOST;
    }

    void move_from(DevicePointer&& other) noexcept
    {
        std::lock_guard<std::mutex> other_lock(other.m_state_mutex);
        m_count = other.m_count;
        m_bytes = other.m_bytes;
        m_host = std::move(other.m_host);
        m_device = std::move(other.m_device);
        m_owner = other.m_owner;
        m_pending_flush = std::move(other.m_pending_flush);

        other.m_count = 0;
        other.m_bytes = 0;
        other.m_owner = DeviceMemoryOwner::INVALID;
        other.m_pending_flush = nullptr;
    }

    void ensure_host_data() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (m_owner == DeviceMemoryOwner::ON_DEVICE)
            copy_from_offload_buffer_unlocked();
        if (m_pending_flush)
        {
            m_pending_flush();
            m_pending_flush = nullptr;
        }
    }

    void ensure_device_data() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        copy_to_offload_buffer_unlocked();
    }

    void mark_host_modified() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_owner = DeviceMemoryOwner::ON_HOST;
        m_pending_flush = nullptr;
    }

    void copy_to_offload_buffer_unlocked(std::string_view site = {}, std::string_view parameter = {}) const
    {
        if (m_owner == DeviceMemoryOwner::ON_HOST || m_owner == DeviceMemoryOwner::INVALID)
        {
            m_device->write(rllm::vulkan_runtime::context(), *m_host);
            parallel::statistics.record_host_to_device_buffer_copy(site, parameter, m_bytes);
            ComputeKernelRegistry::instance().recordHostToDevice(ComputeKernelRegistry::activeKernel(), m_bytes);
            m_owner = DeviceMemoryOwner::REPLICATED;
        }
    }

    void copy_from_offload_buffer_unlocked() const
    {
        if (m_pending_flush)
        {
            m_pending_flush();
            m_pending_flush = nullptr;
        }
        if (m_owner == DeviceMemoryOwner::ON_DEVICE)
        {
            m_device->read(rllm::vulkan_runtime::context(), *m_host);
            parallel::statistics.record_device_to_host_buffer_copy({}, {}, m_bytes);
            ComputeKernelRegistry::instance().recordDeviceToHost(m_last_device_writer, m_bytes);
            m_owner = DeviceMemoryOwner::REPLICATED;
        }
    }

    T* host_data() const { return reinterpret_cast<T*>(host_bytes()); }
    uint8_t* host_bytes() const
    {
        return m_host->bytes();
    }

    size_t m_count = 0;
    size_t m_bytes = 0;
    std::unique_ptr<VDynamicHostBuffer> m_host;
    std::unique_ptr<VDynamicDeviceBuffer> m_device;
    mutable DeviceMemoryOwner m_owner = DeviceMemoryOwner::INVALID;
    mutable std::function<void()> m_pending_flush;
    mutable std::mutex m_state_mutex;
    mutable std::string m_last_device_writer;
};

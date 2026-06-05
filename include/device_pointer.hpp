#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <mutex>
#include <utility>

#include <IMemorySpace.hpp>

#if defined(USE_VULKAN_OFFLOAD)
#include <vulkan/vulkan.h>
#endif

enum class DeviceMemoryOwner {
    INVALID,
    ON_DEVICE,
    ON_HOST,
    REPLICATED,
};

#if defined(USE_VULKAN_OFFLOAD)
struct VulkanRuntimeBuffer
{
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t size_bytes = 0;

    VulkanRuntimeBuffer() = default;
    VulkanRuntimeBuffer(const VulkanRuntimeBuffer&) = delete;
    VulkanRuntimeBuffer& operator=(const VulkanRuntimeBuffer&) = delete;

    VulkanRuntimeBuffer(VulkanRuntimeBuffer&& other) noexcept
    {
        move_from(std::move(other));
    }

    VulkanRuntimeBuffer& operator=(VulkanRuntimeBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    ~VulkanRuntimeBuffer()
    {
        release();
    }

    void release()
    {
        if (device == VK_NULL_HANDLE)
        {
            clear_handles();
            return;
        }

        vkDeviceWaitIdle(device);

        if (mapped != nullptr && staging_memory != VK_NULL_HANDLE)
        {
            vkUnmapMemory(device, staging_memory);
            mapped = nullptr;
        }
        if (staging_buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(device, staging_buffer, nullptr);
        if (staging_memory != VK_NULL_HANDLE)
            vkFreeMemory(device, staging_memory, nullptr);
        if (buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE)
            vkFreeMemory(device, memory, nullptr);

        clear_handles();
    }

private:
    void move_from(VulkanRuntimeBuffer&& other)
    {
        device = other.device;
        buffer = other.buffer;
        memory = other.memory;
        staging_buffer = other.staging_buffer;
        staging_memory = other.staging_memory;
        mapped = other.mapped;
        size_bytes = other.size_bytes;
        other.clear_handles();
    }

    void clear_handles()
    {
        device = VK_NULL_HANDLE;
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        staging_buffer = VK_NULL_HANDLE;
        staging_memory = VK_NULL_HANDLE;
        mapped = nullptr;
        size_bytes = 0;
    }
};
#endif

template <typename T>
class DevicePointer
{
public:
    explicit DevicePointer(size_t num_elements)
        : m_memory_space(IMemorySpace::get_instance())
        , m_count(num_elements)
        , m_bytes(sizeof(T) * num_elements)
    {
        assert(m_count > 0);
        allocate_internal();
        zero();
    }

    DevicePointer(IMemorySpace& memory_space, size_t num_elements)
        : m_memory_space(&memory_space)
        , m_count(num_elements)
        , m_bytes(sizeof(T) * num_elements)
    {
        assert(m_count > 0);
        allocate_internal();
        zero();
    }

    DevicePointer(const DevicePointer& other)
        : m_memory_space(other.m_memory_space != nullptr ? other.m_memory_space : IMemorySpace::get_instance())
        , m_count(other.m_count)
        , m_bytes(other.m_bytes)
    {
        allocate_internal();
        copy_contents_from(other);
    }

    DevicePointer& operator=(const DevicePointer& other)
    {
        if (this == &other)
            return *this;

        assign_storage_like(other);
        copy_contents_from(other);
        return *this;
    }

    DevicePointer(DevicePointer&& other) noexcept
        : m_memory_space(other.m_memory_space)
        , m_count(other.m_count)
        , m_bytes(other.m_bytes)
        , m_staging_ptr(other.m_staging_ptr)
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        , m_offload_ptr(other.m_offload_ptr)
        , m_memory_owner(other.m_memory_owner)
        , m_host_access_fast_path(other.m_host_access_fast_path.load(std::memory_order_acquire))
#if defined(USE_VULKAN_OFFLOAD)
        , m_vulkan_runtime_buffer(std::move(other.m_vulkan_runtime_buffer))
#endif
        , m_pending_flush(std::move(other.m_pending_flush))
#endif
    {
        other.m_memory_space = nullptr;
        other.m_count = 0;
        other.m_bytes = 0;
        other.m_staging_ptr = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        other.m_offload_ptr = nullptr;
        other.m_memory_owner = DeviceMemoryOwner::INVALID;
        other.m_host_access_fast_path.store(false, std::memory_order_release);
        other.m_pending_flush = nullptr;
#endif
    }

    DevicePointer& operator=(DevicePointer&& other) noexcept
    {
        if (this == &other)
            return *this;

        release_internal();
        m_memory_space = other.m_memory_space;
        m_count = other.m_count;
        m_bytes = other.m_bytes;
        m_staging_ptr = other.m_staging_ptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_offload_ptr = other.m_offload_ptr;
        m_memory_owner = other.m_memory_owner;
        m_host_access_fast_path.store(other.m_host_access_fast_path.load(std::memory_order_acquire), std::memory_order_release);
        m_pending_flush = std::move(other.m_pending_flush);
#if defined(USE_VULKAN_OFFLOAD)
        m_vulkan_runtime_buffer = std::move(other.m_vulkan_runtime_buffer);
#endif
#endif

        other.m_memory_space = nullptr;
        other.m_count = 0;
        other.m_bytes = 0;
        other.m_staging_ptr = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        other.m_offload_ptr = nullptr;
        other.m_memory_owner = DeviceMemoryOwner::INVALID;
        other.m_host_access_fast_path.store(false, std::memory_order_release);
        other.m_pending_flush = nullptr;
#endif
        return *this;
    }

    ~DevicePointer()
    {
        release_internal();
    }

    void zero()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        assert(m_bytes > 0);
        assert(m_staging_ptr != nullptr);

        std::memset(m_staging_ptr, 0, m_bytes);
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        assert(m_offload_ptr != nullptr);
        m_memory_space->zero_offload(m_offload_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
        m_pending_flush = nullptr;
        m_host_access_fast_path.store(false, std::memory_order_release);
#endif
    }

    void fill(T value)
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        assert(m_staging_ptr != nullptr);

        std::fill_n(m_staging_ptr, m_count, value);
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        assert(m_offload_ptr != nullptr);
        m_memory_space->copy_staging_to_offload(m_offload_ptr, m_staging_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
        m_pending_flush = nullptr;
        m_host_access_fast_path.store(false, std::memory_order_release);
#endif
    }

    T* get() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        if (host_access_fast_path())
            return m_staging_ptr;
        ensure_host_data();
        mark_host_modified();
#endif
        return m_staging_ptr;
    }

    T* staging_data() const
    {
        return get();
    }

    T* raw_staging_data() const
    {
        return m_staging_ptr;
    }

    void* offload_data() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        ensure_device_data();
        mark_device_modified();
        return m_offload_ptr;
#else
        return nullptr;
#endif
    }

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
    void* raw_offload_data() const
    {
        return m_offload_ptr;
    }
#endif

#if defined(USE_VULKAN_OFFLOAD)
    VulkanRuntimeBuffer& vulkan_runtime_buffer() const
    {
        return m_vulkan_runtime_buffer;
    }
#endif

    size_t size() const
    {
        return m_count;
    }

    size_t storage_size_bytes() const
    {
        return m_bytes;
    }

    void resize(size_t num_elements)
    {
        assert(num_elements > 0);
        if (num_elements == m_count && m_staging_ptr != nullptr)
            return;

        IMemorySpace* target_memory_space = m_memory_space != nullptr ? m_memory_space : IMemorySpace::get_instance();
        DevicePointer resized(*target_memory_space, num_elements);
        const size_t copy_count = std::min(m_count, num_elements);
        if (m_staging_ptr != nullptr && copy_count > 0)
        {
            ensure_host_data();
            std::memcpy(resized.m_staging_ptr, m_staging_ptr, copy_count * sizeof(T));
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
            resized.m_memory_owner = DeviceMemoryOwner::ON_HOST;
            resized.m_host_access_fast_path.store(true, std::memory_order_release);
#endif
        }
        *this = std::move(resized);
    }

    T* data()
    {
        return get();
    }

    const T* data() const
    {
        ensure_host_data();
        return m_staging_ptr;
    }

    void set_pending_flush(std::function<void()> flush_fn)
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_pending_flush = std::move(flush_fn);
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
        m_host_access_fast_path.store(false, std::memory_order_release);
#else
        static_cast<void>(flush_fn);
#endif
    }

    void mark_device_latest()
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_pending_flush = nullptr;
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
        m_host_access_fast_path.store(false, std::memory_order_release);
#endif
    }

    DeviceMemoryOwner device_memory_owner() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return m_memory_owner;
#else
        return DeviceMemoryOwner::ON_HOST;
#endif
    }

    bool has_pending_flush() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return static_cast<bool>(m_pending_flush);
#else
        return false;
#endif
    }

    T& operator[](size_t idx)
    {
        ensure_host_data();
        mark_host_modified();
        return m_staging_ptr[idx];
    }

    const T& operator[](size_t idx) const
    {
        ensure_host_data();
        return m_staging_ptr[idx];
    }

    T get(size_t idx) const
    {
        ensure_host_data();
        return m_staging_ptr[idx];
    }

    T get_offload_synced(size_t idx) const
    {
        assert(idx < m_count);
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        if (device_data_is_current())
        {
            auto* staging_dst = reinterpret_cast<std::byte*>(m_staging_ptr) + idx * sizeof(T);
            const auto* offload_src = static_cast<const std::byte*>(m_offload_ptr) + idx * sizeof(T);
            m_memory_space->copy_offload_to_staging(staging_dst, offload_src, sizeof(T));
            return m_staging_ptr[idx];
        }
#endif
        ensure_host_data();
        return m_staging_ptr[idx];
    }

    void set(size_t idx, const T& value)
    {
        ensure_host_data();
        m_staging_ptr[idx] = value;
        mark_host_modified();
    }

    void copy_to_offload_buffer()
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        copy_to_offload_buffer_unlocked();
#endif
    }

    void copy_range_to_offload_buffer(size_t start_element, size_t element_count)
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        copy_range_to_offload_buffer_unlocked(start_element, element_count);
#else
        static_cast<void>(start_element);
        static_cast<void>(element_count);
#endif
    }

    void copy_from_offload_buffer()
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        copy_from_offload_buffer_unlocked();
#endif
    }

    bool needs_offload_sync() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return m_memory_owner == DeviceMemoryOwner::ON_HOST;
#else
        return false;
#endif
    }

private:
    bool host_data_is_current_unlocked() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        return m_memory_owner == DeviceMemoryOwner::ON_HOST || m_memory_owner == DeviceMemoryOwner::REPLICATED;
#else
        return true;
#endif
    }

    bool device_data_is_current_unlocked() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        return m_memory_owner == DeviceMemoryOwner::ON_DEVICE || m_memory_owner == DeviceMemoryOwner::REPLICATED;
#else
        return false;
#endif
    }

    bool device_data_is_current() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return device_data_is_current_unlocked();
#else
        return false;
#endif
    }

    bool host_access_fast_path() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        return m_host_access_fast_path.load(std::memory_order_acquire);
#else
        return false;
#endif
    }

    void mark_host_modified() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        if (m_host_access_fast_path.load(std::memory_order_acquire))
            return;

        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        self->m_pending_flush = nullptr;
        self->m_memory_owner = DeviceMemoryOwner::ON_HOST;
        self->m_host_access_fast_path.store(true, std::memory_order_release);
#endif
    }

    void mark_device_modified() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        self->m_pending_flush = nullptr;
        self->m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
        self->m_host_access_fast_path.store(false, std::memory_order_release);
#endif
    }

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
    void copy_to_offload_buffer_unlocked()
    {
        assert(m_staging_ptr != nullptr);
        assert(m_offload_ptr != nullptr);

        if (device_data_is_current_unlocked() && m_memory_owner != DeviceMemoryOwner::ON_HOST)
            return;

        m_memory_space->copy_staging_to_offload(m_offload_ptr, m_staging_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::REPLICATED;
        m_host_access_fast_path.store(false, std::memory_order_release);
    }

    void copy_range_to_offload_buffer_unlocked(size_t start_element, size_t element_count)
    {
        assert(m_staging_ptr != nullptr);
        assert(m_offload_ptr != nullptr);
        assert(start_element <= m_count);
        assert(element_count <= m_count - start_element);

        if (element_count == 0)
            return;

        const size_t byte_offset = start_element * sizeof(T);
        const size_t bytes = element_count * sizeof(T);
        auto* offload_dst = static_cast<std::byte*>(m_offload_ptr) + byte_offset;
        const auto* staging_src = reinterpret_cast<const std::byte*>(m_staging_ptr) + byte_offset;
        m_memory_space->copy_staging_to_offload(offload_dst, staging_src, bytes);
        m_memory_owner = DeviceMemoryOwner::REPLICATED;
        m_host_access_fast_path.store(false, std::memory_order_release);
    }

    void copy_from_offload_buffer_unlocked()
    {
        assert(m_staging_ptr != nullptr);
        assert(m_offload_ptr != nullptr);

        if (host_data_is_current_unlocked() && m_memory_owner != DeviceMemoryOwner::ON_DEVICE)
            return;

        m_memory_space->copy_offload_to_staging(m_staging_ptr, m_offload_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::REPLICATED;
        m_host_access_fast_path.store(false, std::memory_order_release);
    }
#endif

    void assign_storage_like(const DevicePointer& other)
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        if (m_count == other.m_count && m_bytes == other.m_bytes && m_staging_ptr != nullptr && m_offload_ptr != nullptr)
#else
        if (m_count == other.m_count && m_bytes == other.m_bytes && m_staging_ptr != nullptr)
#endif
        {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
            m_pending_flush = nullptr;
            m_host_access_fast_path.store(false, std::memory_order_release);
#endif
            return;
        }

        release_internal();
        m_memory_space = other.m_memory_space != nullptr ? other.m_memory_space : IMemorySpace::get_instance();
        m_count = other.m_count;
        m_bytes = other.m_bytes;
        allocate_internal();
    }

    void copy_contents_from(const DevicePointer& other)
    {
        assert(m_bytes != 0);

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_pending_flush = nullptr;
        m_host_access_fast_path.store(false, std::memory_order_release);

        switch (other.m_memory_owner)
        {
            case DeviceMemoryOwner::INVALID:
                std::abort();
            case DeviceMemoryOwner::ON_HOST:
                std::memcpy(m_staging_ptr, other.m_staging_ptr, m_bytes);
                m_memory_owner = DeviceMemoryOwner::ON_HOST;
                m_host_access_fast_path.store(true, std::memory_order_release);
                break;
            case DeviceMemoryOwner::ON_DEVICE:
                m_memory_space->copy_offload_to_offload(m_offload_ptr, other.m_offload_ptr, m_bytes);
                m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
                break;
            case DeviceMemoryOwner::REPLICATED:
                std::memcpy(m_staging_ptr, other.m_staging_ptr, m_bytes);
                m_memory_space->copy_offload_to_offload(m_offload_ptr, other.m_offload_ptr, m_bytes);
                m_memory_owner = DeviceMemoryOwner::REPLICATED;
                break;
        }
#else
        std::memcpy(m_staging_ptr, other.m_staging_ptr, m_bytes);
#endif
    }

    void allocate_internal()
    {
        if (m_memory_space == nullptr)
            m_memory_space = IMemorySpace::get_instance();
        if (m_bytes == 0)
            return;

        m_staging_ptr = static_cast<T*>(m_memory_space->allocate_staging(m_bytes));
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_offload_ptr = m_memory_space->allocate_offload(m_bytes);
        if (m_staging_ptr == nullptr || m_offload_ptr == nullptr)
#else
        if (m_staging_ptr == nullptr)
#endif
        {
            release_internal();
            std::abort();
        }

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_memory_owner = DeviceMemoryOwner::INVALID;
        m_pending_flush = nullptr;
        m_host_access_fast_path.store(false, std::memory_order_release);
#endif
    }

    void ensure_host_data() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        if (!self->host_data_is_current_unlocked() && self->device_data_is_current_unlocked())
        {
            if (self->m_pending_flush)
            {
                auto fn = std::move(self->m_pending_flush);
                self->m_pending_flush = nullptr;
                fn();
                self->m_memory_owner = DeviceMemoryOwner::REPLICATED;
                self->m_host_access_fast_path.store(false, std::memory_order_release);
            }
            else
            {
                self->copy_from_offload_buffer_unlocked();
            }
        }
#endif
    }

    void ensure_device_data() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        if (!self->device_data_is_current_unlocked() && self->host_data_is_current_unlocked())
            self->copy_to_offload_buffer_unlocked();
#endif
    }

    void release_internal()
    {
        if (m_memory_space != nullptr && m_staging_ptr != nullptr)
            m_memory_space->release_staging(m_staging_ptr, m_bytes);

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_pending_flush = nullptr;
        m_host_access_fast_path.store(false, std::memory_order_release);
#if defined(USE_VULKAN_OFFLOAD)
        m_vulkan_runtime_buffer.release();
#endif
        if (m_memory_space != nullptr && m_offload_ptr != nullptr)
            m_memory_space->release_offload(m_offload_ptr, m_bytes);
        m_offload_ptr = nullptr;
        m_memory_owner = DeviceMemoryOwner::INVALID;
#endif

        m_staging_ptr = nullptr;
    }

    IMemorySpace* m_memory_space = nullptr;
    size_t m_count = 0;
    size_t m_bytes = 0;
    T* m_staging_ptr = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
    void* m_offload_ptr = nullptr;
    mutable DeviceMemoryOwner m_memory_owner = DeviceMemoryOwner::INVALID;
    mutable std::atomic_bool m_host_access_fast_path{false};
#if defined(USE_VULKAN_OFFLOAD)
    mutable VulkanRuntimeBuffer m_vulkan_runtime_buffer;
#endif
    std::function<void()> m_pending_flush;
#endif
    mutable std::mutex m_state_mutex;
};

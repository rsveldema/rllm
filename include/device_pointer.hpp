#pragma once

#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>
#include <cassert>

#include <IMemorySpace.hpp>
#include <allocator.hpp>

Allocator& get_offload_allocator();

enum class DeviceMemoryOwner {
    INVALID,
    ON_DEVICE,
    ON_HOST,
    REPLICATED,
};

template<typename T>
class DevicePointer
{
public:
    static constexpr size_t kAlignment = alignof(T);

    explicit DevicePointer(size_t num_elements)
        : m_allocator(&get_offload_allocator())
        , m_memory_space(&m_allocator->memory_space())
        , m_count(num_elements)
    {
        assert(m_count > 0);

        allocate_internal();
        zero();
        m_memory_owner = DeviceMemoryOwner::REPLICATED;
        assert(m_bytes > 0);
    }

    DevicePointer(Allocator& allocator, size_t num_elements)
        : m_allocator(&allocator)
        , m_memory_space(&allocator.memory_space())
        , m_count(num_elements)
    {
        assert(m_count > 0);

        allocate_internal();
        zero();
        m_memory_owner = DeviceMemoryOwner::REPLICATED;

        assert(m_bytes > 0);
    }


    DevicePointer(const DevicePointer& other)
        : m_allocator(&get_offload_allocator())
        , m_memory_space(&m_allocator->memory_space())
        , m_count(other.m_count)
        , m_bytes(other.m_bytes)
    {
        assert(m_bytes > 0);

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, kAlignment));
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, kAlignment);
        if (m_staging_ptr == nullptr || m_offload_ptr == nullptr)
        {
            release_internal();
            std::abort();
        }

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
        : m_allocator(other.m_allocator)
        , m_memory_space(other.m_memory_space)
        , m_count(other.m_count)
        , m_bytes(other.m_bytes)
        , m_staging_ptr(other.m_staging_ptr)
        , m_offload_ptr(other.m_offload_ptr)
        , m_memory_owner(other.m_memory_owner)
        , m_pending_flush(std::move(other.m_pending_flush))
    {
        other.m_allocator = nullptr;
        other.m_memory_space = nullptr;
        other.m_count = 0;
        other.m_bytes = 0;
        other.m_staging_ptr = nullptr;
        other.m_offload_ptr = nullptr;
        other.m_memory_owner = DeviceMemoryOwner::INVALID;
        other.m_pending_flush = nullptr;

        assert(m_bytes > 0);
    }

    DevicePointer& operator=(DevicePointer&& other) noexcept
    {
        if (this == &other)
            return *this;

        release_internal();

        m_allocator = other.m_allocator;
        m_memory_space = other.m_memory_space;
        m_count = other.m_count;
        m_bytes = other.m_bytes;
        m_staging_ptr = other.m_staging_ptr;
        m_offload_ptr = other.m_offload_ptr;
        m_memory_owner = other.m_memory_owner;
        m_pending_flush = std::move(other.m_pending_flush);

        other.m_allocator = nullptr;
        other.m_memory_space = nullptr;
        other.m_count = 0;
        other.m_bytes = 0;
        other.m_staging_ptr = nullptr;
        other.m_offload_ptr = nullptr;
        other.m_memory_owner = DeviceMemoryOwner::INVALID;
        other.m_pending_flush = nullptr;

        assert(m_bytes > 0);
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
        assert(m_offload_ptr != nullptr);

        zero_initialize_staging();
        zero_initialize_offload();
        m_memory_owner = DeviceMemoryOwner::REPLICATED;
        m_pending_flush = nullptr;
    }

    T* get() const
    {
        ensure_host_data();
        mark_host_modified();
        return m_staging_ptr;
    }

    T* staging_data() const
    {
        ensure_host_data();
        mark_host_modified();
        return m_staging_ptr;
    }

    T* raw_staging_data() const
    {
        return m_staging_ptr;
    }

    void* raw_offload_data() const
    {
        return m_offload_ptr;
    }

    void* offload_data() const
    {
        ensure_device_data();
        mark_device_modified();
        return m_offload_ptr;
    }

    size_t size() const
    {
        return m_count;
    }

    size_t storage_size_bytes() const
    {
        return m_bytes;
    }

    T* data()
    {
        ensure_host_data();
        mark_host_modified();
        return m_staging_ptr;
    }

    const T* data() const
    {
        ensure_host_data();
        return m_staging_ptr;
    }

    void set_pending_flush(std::function<void()> flush_fn)
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_pending_flush = std::move(flush_fn);
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
    }

    void mark_device_latest()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_pending_flush = nullptr;
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
    }

    DeviceMemoryOwner device_memory_owner() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return m_memory_owner;
    }

    bool has_pending_flush() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return static_cast<bool>(m_pending_flush);
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

    void set(size_t idx, const T& value)
    {
        ensure_host_data();
        m_staging_ptr[idx] = value;
        mark_host_modified();
    }

    void copy_to_offload_buffer()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        copy_to_offload_buffer_unlocked();
    }

    void copy_from_offload_buffer()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        copy_from_offload_buffer_unlocked();
    }

    bool needs_offload_sync() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return m_memory_owner == DeviceMemoryOwner::ON_HOST;
    }

private:
    bool host_data_is_current_unlocked() const
    {
        return m_memory_owner == DeviceMemoryOwner::ON_HOST || m_memory_owner == DeviceMemoryOwner::REPLICATED;
    }

    bool device_data_is_current_unlocked() const
    {
        return m_memory_owner == DeviceMemoryOwner::ON_DEVICE || m_memory_owner == DeviceMemoryOwner::REPLICATED;
    }

    void mark_host_modified() const
    {
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        self->m_pending_flush = nullptr;
        self->m_memory_owner = DeviceMemoryOwner::ON_HOST;
    }

    void mark_device_modified() const
    {
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        self->m_pending_flush = nullptr;
        self->m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
    }

    void copy_to_offload_buffer_unlocked()
    {
        assert(m_bytes > 0);
        assert(m_staging_ptr != nullptr);
        assert(m_offload_ptr != nullptr);

        if (device_data_is_current_unlocked() && m_memory_owner != DeviceMemoryOwner::ON_HOST)
            return;

        m_memory_space->copy_staging_to_offload(m_offload_ptr, m_staging_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::REPLICATED;
    }

    void copy_from_offload_buffer_unlocked()
    {
        assert(m_bytes > 0);
        assert(m_staging_ptr != nullptr);
        assert(m_offload_ptr != nullptr);

        if (host_data_is_current_unlocked() && m_memory_owner != DeviceMemoryOwner::ON_DEVICE)
            return;

        m_memory_space->copy_offload_to_staging(m_staging_ptr, m_offload_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::REPLICATED;
    }

    void assign_storage_like(const DevicePointer& other)
    {
        if (m_count == other.m_count && m_bytes == other.m_bytes &&
            m_staging_ptr != nullptr && m_offload_ptr != nullptr)
        {
            m_pending_flush = nullptr;
            return;
        }

        release_internal();

        m_allocator = &get_offload_allocator();
        m_memory_space = &m_allocator->memory_space();
        m_count = other.m_count;
        m_bytes = other.m_bytes;
        if (m_bytes == 0)
        {
            m_memory_owner = DeviceMemoryOwner::INVALID;
            m_pending_flush = nullptr;
            return;
        }

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, kAlignment));
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, kAlignment);
        if (m_staging_ptr == nullptr || m_offload_ptr == nullptr)
        {
            release_internal();
            std::abort();
        }

        m_memory_owner = DeviceMemoryOwner::INVALID;
        m_pending_flush = nullptr;
    }

    void copy_host_to_host(const DevicePointer& other)
    {
        assert(other.m_staging_ptr != nullptr);
        assert(m_staging_ptr != nullptr);

        std::memcpy(m_staging_ptr, other.m_staging_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::ON_HOST;
    }

    void copy_device_to_device(const DevicePointer& other)
    {
        assert(other.m_offload_ptr != nullptr);
        assert(m_offload_ptr != nullptr);

        m_memory_space->copy_offload_to_offload(m_offload_ptr, other.m_offload_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
    }

    void copy_replicated_to_replicated(const DevicePointer& other)
    {
        assert(other.m_staging_ptr != nullptr);
        assert(other.m_offload_ptr != nullptr);
        assert(m_staging_ptr != nullptr);
        assert(m_offload_ptr != nullptr);

        std::memcpy(m_staging_ptr, other.m_staging_ptr, m_bytes);
        m_memory_space->copy_offload_to_offload(m_offload_ptr, other.m_offload_ptr, m_bytes);
        m_memory_owner = DeviceMemoryOwner::REPLICATED;
    }

    void copy_contents_from(const DevicePointer& other)
    {
        m_pending_flush = nullptr;

        assert(m_bytes != 0);

        switch (other.m_memory_owner)
        {
            case DeviceMemoryOwner::INVALID:
                std::abort();

            case DeviceMemoryOwner::ON_HOST:
                copy_host_to_host(other);
                break;

            case DeviceMemoryOwner::ON_DEVICE:
                copy_device_to_device(other);
                break;

            case DeviceMemoryOwner::REPLICATED:
                copy_replicated_to_replicated(other);
                break;
        }
    }

    void allocate_internal()
    {
        if (m_allocator == nullptr)
            m_allocator = &get_offload_allocator();
        m_memory_space = &m_allocator->memory_space();
        m_bytes = sizeof(T) * m_count;
        if (m_bytes == 0)
            return;

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, kAlignment));
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, kAlignment);
        if (m_staging_ptr == nullptr || m_offload_ptr == nullptr)
        {
            release_internal();
            std::abort();
        }

        m_memory_owner = DeviceMemoryOwner::INVALID;
        m_pending_flush = nullptr;
    }

    void zero_initialize_staging()
    {
        assert(m_staging_ptr != nullptr);
        std::memset((void*)m_staging_ptr, 0, m_bytes);
    }

    void zero_initialize_offload()
    {
        assert(m_offload_ptr != nullptr);

        m_memory_space->zero_offload(m_offload_ptr, m_bytes);
        m_pending_flush = nullptr;
    }

    void ensure_host_data() const
    {
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
            }
            else
            {
                self->copy_from_offload_buffer_unlocked();
            }
        }
    }

    void ensure_device_data() const
    {
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        if (!self->device_data_is_current_unlocked() && self->host_data_is_current_unlocked())
            self->copy_to_offload_buffer_unlocked();
    }

    void release_internal()
    {
        m_pending_flush = nullptr;

        if (m_staging_ptr != nullptr)
            m_allocator->release_staging(m_staging_ptr, kAlignment);

        if (m_offload_ptr != nullptr)
            m_allocator->release_offload(m_offload_ptr, kAlignment);

        m_staging_ptr = nullptr;
        m_offload_ptr = nullptr;
        m_memory_owner = DeviceMemoryOwner::INVALID;
    }

    Allocator* m_allocator = nullptr;
    IMemorySpace* m_memory_space = nullptr;
    size_t m_count = 0;
    size_t m_bytes = 0;
    T* m_staging_ptr = nullptr;
    void* m_offload_ptr = nullptr;
    mutable DeviceMemoryOwner m_memory_owner = DeviceMemoryOwner::INVALID;
    mutable std::mutex m_state_mutex;
    std::function<void()> m_pending_flush;
};

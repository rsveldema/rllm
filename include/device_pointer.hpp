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

#if defined(USE_VULKAN_OFFLOAD) || defined(USE_HIP_OFFLOAD)
#define RLLM_DEVICE_POINTER_HAS_OFFLOAD 1
#else
#define RLLM_DEVICE_POINTER_HAS_OFFLOAD 0
#endif

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
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        , m_memory_space(&m_allocator->memory_space())
#endif
        , m_count(num_elements)
    {
        assert(m_count > 0);

        allocate_internal();
        zero();
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
#endif
        assert(m_bytes > 0);
    }

    DevicePointer(Allocator& allocator, size_t num_elements)
        : m_allocator(&allocator)
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        , m_memory_space(&allocator.memory_space())
#endif
        , m_count(num_elements)
    {
        assert(m_count > 0);

        allocate_internal();
        zero();
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
#endif

        assert(m_bytes > 0);
    }


    DevicePointer(const DevicePointer& other)
        : m_allocator(&get_offload_allocator())
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        , m_memory_space(&m_allocator->memory_space())
#endif
        , m_count(other.m_count)
        , m_bytes(other.m_bytes)
    {
        assert(m_bytes > 0);

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, kAlignment));
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, kAlignment);
        if (m_staging_ptr == nullptr || m_offload_ptr == nullptr)
#else
        if (m_staging_ptr == nullptr)
#endif
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
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        , m_memory_space(other.m_memory_space)
#endif
        , m_count(other.m_count)
        , m_bytes(other.m_bytes)
        , m_staging_ptr(other.m_staging_ptr)
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        , m_offload_ptr(other.m_offload_ptr)
        , m_memory_owner(other.m_memory_owner)
        , m_pending_flush(std::move(other.m_pending_flush))
#endif
    {
        other.m_allocator = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        other.m_memory_space = nullptr;
#endif
        other.m_count = 0;
        other.m_bytes = 0;
        other.m_staging_ptr = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        other.m_offload_ptr = nullptr;
        other.m_memory_owner = DeviceMemoryOwner::INVALID;
        other.m_pending_flush = nullptr;
#endif

        assert(m_bytes > 0);
    }

    DevicePointer& operator=(DevicePointer&& other) noexcept
    {
        if (this == &other)
            return *this;

        release_internal();

        m_allocator = other.m_allocator;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_memory_space = other.m_memory_space;
#endif
        m_count = other.m_count;
        m_bytes = other.m_bytes;
        m_staging_ptr = other.m_staging_ptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_offload_ptr = other.m_offload_ptr;
        m_memory_owner = other.m_memory_owner;
        m_pending_flush = std::move(other.m_pending_flush);
#endif

        other.m_allocator = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        other.m_memory_space = nullptr;
#endif
        other.m_count = 0;
        other.m_bytes = 0;
        other.m_staging_ptr = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        other.m_offload_ptr = nullptr;
        other.m_memory_owner = DeviceMemoryOwner::INVALID;
        other.m_pending_flush = nullptr;
#endif

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
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        assert(m_offload_ptr != nullptr);
#endif

        zero_initialize_staging();
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        zero_initialize_offload();
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
        m_pending_flush = nullptr;
#endif
    }

    void fill(T value)
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        assert(m_bytes > 0);
        assert(m_staging_ptr != nullptr);
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        assert(m_offload_ptr != nullptr);
#endif

        fill_staging(value);
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        fill_offload(value);
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
        m_pending_flush = nullptr;
#endif
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
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        return m_offload_ptr;
#else
        return nullptr;
#endif
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
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_pending_flush = std::move(flush_fn);
        m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
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

    void mark_host_modified() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        self->m_pending_flush = nullptr;
        self->m_memory_owner = DeviceMemoryOwner::ON_HOST;
#endif
    }

    void mark_device_modified() const
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        self->m_pending_flush = nullptr;
        self->m_memory_owner = DeviceMemoryOwner::ON_DEVICE;
#endif
    }

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
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
#endif

    void assign_storage_like(const DevicePointer& other)
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        if (m_count == other.m_count && m_bytes == other.m_bytes &&
            m_staging_ptr != nullptr && m_offload_ptr != nullptr)
#else
        if (m_count == other.m_count && m_bytes == other.m_bytes && m_staging_ptr != nullptr)
#endif
        {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
            m_pending_flush = nullptr;
#endif
            return;
        }

        release_internal();

        m_allocator = &get_offload_allocator();
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_memory_space = &m_allocator->memory_space();
#endif
        m_count = other.m_count;
        m_bytes = other.m_bytes;
        if (m_bytes == 0)
        {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
            m_memory_owner = DeviceMemoryOwner::INVALID;
            m_pending_flush = nullptr;
#endif
            return;
        }

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, kAlignment));
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, kAlignment);
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
#endif
    }

    void copy_host_to_host(const DevicePointer& other)
    {
        assert(other.m_staging_ptr != nullptr);
        assert(m_staging_ptr != nullptr);

        std::memcpy(m_staging_ptr, other.m_staging_ptr, m_bytes);
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_memory_owner = DeviceMemoryOwner::ON_HOST;
#endif
    }

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
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
#endif

    void copy_contents_from(const DevicePointer& other)
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_pending_flush = nullptr;
#endif

        assert(m_bytes != 0);

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
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
#else
        copy_host_to_host(other);
#endif
    }

    void allocate_internal()
    {
        if (m_allocator == nullptr)
            m_allocator = &get_offload_allocator();
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_memory_space = &m_allocator->memory_space();
#endif
        m_bytes = sizeof(T) * m_count;
        if (m_bytes == 0)
            return;

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, kAlignment));
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, kAlignment);
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
#endif
    }

    void fill_staging(T value)
    {
        assert(m_staging_ptr != nullptr);
        std::fill_n(m_staging_ptr, m_count, value);
    }

    void fill_offload(T value)
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        (void)value;
        assert(m_staging_ptr != nullptr);
        assert(m_offload_ptr != nullptr);

        m_memory_space->copy_staging_to_offload(m_offload_ptr, m_staging_ptr, m_bytes);
#else
        static_cast<void>(value);
#endif
    }

    void zero_initialize_staging()
    {
        assert(m_staging_ptr != nullptr);
        std::memset(m_staging_ptr, 0, m_bytes);
    }

    void zero_initialize_offload()
    {
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        assert(m_offload_ptr != nullptr);

        m_memory_space->zero_offload(m_offload_ptr, m_bytes);
        m_pending_flush = nullptr;
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
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_pending_flush = nullptr;
#endif

        if (m_staging_ptr != nullptr)
            m_allocator->release_staging(m_staging_ptr, kAlignment);

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        if (m_offload_ptr != nullptr)
            m_allocator->release_offload(m_offload_ptr, kAlignment);
#endif

        m_staging_ptr = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        m_offload_ptr = nullptr;
        m_memory_owner = DeviceMemoryOwner::INVALID;
#endif
    }

    Allocator* m_allocator = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
    IMemorySpace* m_memory_space = nullptr;
#endif
    size_t m_count = 0;
    size_t m_bytes = 0;
    T* m_staging_ptr = nullptr;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
    void* m_offload_ptr = nullptr;
    mutable DeviceMemoryOwner m_memory_owner = DeviceMemoryOwner::INVALID;
#endif
    mutable std::mutex m_state_mutex;
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
    std::function<void()> m_pending_flush;
#endif
};

#undef RLLM_DEVICE_POINTER_HAS_OFFLOAD

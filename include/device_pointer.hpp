#pragma once

#include <cstddef>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

#include <IMemorySpace.hpp>
#include <allocator.hpp>

Allocator& get_offload_allocator();

template<typename T>
class DevicePointer
{
public:
    enum class CurrentOwner
    {
        Device,
        Host,
    };

    explicit DevicePointer(size_t num_elements)
        : m_allocator(&get_offload_allocator())
        , m_memory_space(&m_allocator->memory_space())
        , m_count(num_elements)
    {
        allocate_internal();
        zero_initialize_staging();
        zero_initialize_offload();
    }

    DevicePointer(Allocator& allocator, size_t num_elements)
        : m_allocator(&allocator)
        , m_memory_space(&allocator.memory_space())
        , m_count(num_elements)
        , m_bytes(sizeof(T) * num_elements)
        , m_alignment(alignof(T))
        , m_staging_ptr(static_cast<T*>(allocator.allocate_staging(m_bytes, m_alignment)))
        , m_offload_ptr(allocator.allocate_offload(m_bytes, m_alignment))
        , m_owns_staging(false)
        , m_owns_offload(false)
        , m_owns_heap_raw(false)
        , m_current_owner(CurrentOwner::Host)
    {
        zero_initialize_staging();
        zero_initialize_offload();
    }

    DevicePointer(T* ptr, size_t num_elements, bool owns_heap = false)
        : m_allocator(nullptr)
        , m_memory_space(IMemorySpace::get_instance())
        , m_count(num_elements)
        , m_bytes(sizeof(T) * num_elements)
        , m_alignment(alignof(T))
        , m_staging_ptr(ptr)
        , m_offload_ptr(ptr)
        , m_owns_staging(owns_heap)
        , m_owns_offload(false)
        , m_owns_heap_raw(owns_heap)
        , m_current_owner(CurrentOwner::Host)
    {}

    DevicePointer(const DevicePointer& other)
        : m_allocator(&get_offload_allocator())
        , m_memory_space(&m_allocator->memory_space())
        , m_count(other.m_count)
        , m_bytes(other.m_bytes)
        , m_alignment(other.m_alignment)
        , m_owns_staging(false)
        , m_owns_offload(false)
        , m_owns_heap_raw(false)
        , m_current_owner(CurrentOwner::Host)
    {
        if (m_bytes == 0)
            return;

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, m_alignment));
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, m_alignment);
        if (m_staging_ptr == nullptr || m_offload_ptr == nullptr)
        {
            release_internal();
            throw std::bad_alloc();
        }

        copy_contents_from(other);
    }

    void zero()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (m_bytes == 0 || m_staging_ptr == nullptr)
            return;

        if (m_current_owner == CurrentOwner::Device)
        {
            if (m_offload_ptr != nullptr)
            {
                m_memory_space->zero_offload(m_offload_ptr, m_bytes);
                m_pending_flush = nullptr;
                m_current_owner = CurrentOwner::Device;
                return;
            }
        }

        zero_initialize_staging();
        m_pending_flush = nullptr;
        m_current_owner = CurrentOwner::Host;
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
        , m_alignment(other.m_alignment)
        , m_staging_ptr(other.m_staging_ptr)
        , m_offload_ptr(other.m_offload_ptr)
        , m_owns_staging(other.m_owns_staging)
        , m_owns_offload(other.m_owns_offload)
        , m_owns_heap_raw(other.m_owns_heap_raw)
        , m_current_owner(other.m_current_owner)
        , m_pending_flush(std::move(other.m_pending_flush))
    {
        other.m_allocator = nullptr;
        other.m_memory_space = nullptr;
        other.m_count = 0;
        other.m_bytes = 0;
        other.m_alignment = alignof(T);
        other.m_staging_ptr = nullptr;
        other.m_offload_ptr = nullptr;
        other.m_owns_staging = false;
        other.m_owns_offload = false;
        other.m_owns_heap_raw = false;
        other.m_current_owner = CurrentOwner::Host;
        other.m_pending_flush = nullptr;
    }

    DevicePointer& operator=(DevicePointer&& other) noexcept
    {
        if (this == &other)
            return *this;

        assign_storage_like(other);
        copy_contents_from(other);
        return *this;
    }

    ~DevicePointer()
    {
        release_internal();
    }

    T* get() const
    {
        ensure_host_data();
        m_current_owner = CurrentOwner::Host;
        return m_staging_ptr;
    }

    T* staging_data() const
    {
        ensure_host_data();
        m_current_owner = CurrentOwner::Host;
        return m_staging_ptr;
    }

    T* raw_staging_data() const
    {
        return m_staging_ptr;
    }

    void* offload_data() const
    {
        ensure_device_data();
        m_current_owner = CurrentOwner::Device;
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
        m_current_owner = CurrentOwner::Host;
        return m_staging_ptr;
    }

    const T* data() const
    {
        ensure_host_data();
        return m_staging_ptr;
    }

    // Called by Vulkan kernel infrastructure to register a lazy flush callback.
    // When set, ensure_host_data() will invoke the callback instead of copy_from_offload_buffer().
    void set_pending_flush(std::function<void()> flush_fn)
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_pending_flush = std::move(flush_fn);
        m_current_owner = CurrentOwner::Device;
    }

    bool has_pending_flush() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return static_cast<bool>(m_pending_flush);
    }

    T& operator[](size_t idx)
    {
        ensure_host_data();
        m_current_owner = CurrentOwner::Host;
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
        m_current_owner = CurrentOwner::Host;
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
        return m_current_owner == CurrentOwner::Host;
    }

private:
    void copy_to_offload_buffer_unlocked()
    {
        if (m_bytes == 0 || m_offload_ptr == nullptr || m_staging_ptr == nullptr)
            return;

        if (m_current_owner == CurrentOwner::Device)
            return;

        m_memory_space->copy_staging_to_offload(m_offload_ptr, m_staging_ptr, m_bytes);
        m_current_owner = CurrentOwner::Device;
    }

    void copy_from_offload_buffer_unlocked()
    {
        if (m_bytes == 0 || m_offload_ptr == nullptr || m_staging_ptr == nullptr)
            return;

        if (m_current_owner == CurrentOwner::Host)
            return;

        m_memory_space->copy_offload_to_staging(m_staging_ptr, m_offload_ptr, m_bytes);
        m_current_owner = CurrentOwner::Host;
    }

    void assign_storage_like(const DevicePointer& other)
    {
        if (m_count == other.m_count && m_bytes == other.m_bytes && m_alignment == other.m_alignment &&
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
        m_alignment = other.m_alignment;
        if (m_bytes == 0)
        {
            m_current_owner = CurrentOwner::Host;
            m_pending_flush = nullptr;
            return;
        }

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, m_alignment));
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, m_alignment);
        if (m_staging_ptr == nullptr || m_offload_ptr == nullptr)
        {
            release_internal();
            throw std::bad_alloc();
        }

        m_owns_staging = false;
        m_owns_offload = false;
        m_owns_heap_raw = false;
        m_current_owner = CurrentOwner::Host;
        m_pending_flush = nullptr;
    }

    void copy_contents_from(const DevicePointer& other)
    {
        m_pending_flush = nullptr;

        if (m_bytes == 0)
        {
            m_current_owner = CurrentOwner::Host;
            return;
        }

        const bool can_copy_device_to_device =
            [this, &other]() {
                std::lock_guard<std::mutex> other_lock(other.m_state_mutex);
                return m_memory_space == other.m_memory_space &&
                    other.m_current_owner == CurrentOwner::Device &&
                    !other.m_pending_flush;
            }();

        if (can_copy_device_to_device)
        {
            m_memory_space->copy_offload_to_offload(m_offload_ptr, other.m_offload_ptr, m_bytes);
            m_current_owner = CurrentOwner::Device;
            return;
        }

        other.ensure_host_data();
        std::memcpy(m_staging_ptr, other.m_staging_ptr, m_bytes);
        copy_to_offload_buffer();
        m_current_owner = CurrentOwner::Host;
    }

    void allocate_internal()
    {
        if (m_allocator == nullptr)
            m_allocator = &get_offload_allocator();
        m_memory_space = &m_allocator->memory_space();
        m_bytes = sizeof(T) * m_count;
        m_alignment = alignof(T);
        if (m_bytes == 0)
            return;

        m_staging_ptr = static_cast<T*>(m_allocator->allocate_staging(m_bytes, m_alignment));
        m_offload_ptr = m_allocator->allocate_offload(m_bytes, m_alignment);
        if (m_staging_ptr == nullptr || m_offload_ptr == nullptr)
        {
            release_internal();
            throw std::bad_alloc();
        }
        m_owns_staging = false;
        m_owns_offload = false;
        m_owns_heap_raw = false;
        m_current_owner = CurrentOwner::Host;
        m_pending_flush = nullptr;
    }

    void zero_initialize_staging()
    {
        if (m_staging_ptr == nullptr)
            return;

        if constexpr (std::is_trivially_copyable_v<T>)
        {
            std::memset(m_staging_ptr, 0, m_bytes);
        }
        else
        {
            for (size_t i = 0; i < m_count; ++i)
                m_staging_ptr[i] = T{};
        }
        m_current_owner = CurrentOwner::Host;
    }

    void zero_initialize_offload()
    {
        if (m_offload_ptr == nullptr || m_bytes == 0)
            return;

        m_memory_space->zero_offload(m_offload_ptr, m_bytes);
        m_pending_flush = nullptr;
        m_current_owner = CurrentOwner::Device;
    }

    void ensure_host_data() const
    {
        auto* self = const_cast<DevicePointer*>(this);
        std::lock_guard<std::mutex> lock(self->m_state_mutex);
        if (self->m_current_owner == CurrentOwner::Device)
        {
            if (self->m_pending_flush)
            {
                auto fn = std::move(self->m_pending_flush);
                self->m_pending_flush = nullptr;
                fn();
                self->m_current_owner = CurrentOwner::Host;
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
        if (self->m_current_owner == CurrentOwner::Host)
        {
            self->copy_to_offload_buffer_unlocked();
        }
    }

    void release_internal()
    {
        if (m_owns_heap_raw && m_staging_ptr != nullptr)
            delete[] m_staging_ptr;

        m_staging_ptr = nullptr;
        m_offload_ptr = nullptr;
        m_owns_staging = false;
        m_owns_offload = false;
        m_owns_heap_raw = false;
        m_current_owner = CurrentOwner::Host;
    }

    Allocator* m_allocator = nullptr;
    IMemorySpace* m_memory_space = nullptr;
    size_t m_count = 0;
    size_t m_bytes = 0;
    size_t m_alignment = alignof(T);
    T* m_staging_ptr = nullptr;
    void* m_offload_ptr = nullptr;
    bool m_owns_staging = false;
    bool m_owns_offload = false;
    bool m_owns_heap_raw = false;
    mutable CurrentOwner m_current_owner = CurrentOwner::Host;
    mutable std::mutex m_state_mutex;
    std::function<void()> m_pending_flush;
};

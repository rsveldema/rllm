#pragma once

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

template<typename T>
class DevicePointer
{
public:
    explicit DevicePointer(size_t num_elements)
        : m_allocator(&get_offload_allocator())
        , m_memory_space(&m_allocator->memory_space())
        , m_count(num_elements)
    {
        allocate_internal();
        zero_initialize_staging();
        copy_to_offload_buffer();
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
        , m_staging_dirty(true)
    {
        zero_initialize_staging();
        copy_to_offload_buffer();
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
        , m_staging_dirty(false)
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
        , m_staging_dirty(true)
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

        std::memcpy(m_staging_ptr, other.m_staging_ptr, m_bytes);
        copy_to_offload_buffer();
    }

    DevicePointer& operator=(const DevicePointer& other)
    {
        if (this == &other)
            return *this;

        DevicePointer tmp(other);
        swap(tmp);
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
        , m_staging_dirty(other.m_staging_dirty)
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
        other.m_staging_dirty = false;
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
        m_alignment = other.m_alignment;
        m_staging_ptr = other.m_staging_ptr;
        m_offload_ptr = other.m_offload_ptr;
        m_owns_staging = other.m_owns_staging;
        m_owns_offload = other.m_owns_offload;
        m_owns_heap_raw = other.m_owns_heap_raw;
        m_staging_dirty = other.m_staging_dirty;

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
        other.m_staging_dirty = false;
        return *this;
    }

    ~DevicePointer()
    {
        release_internal();
    }

    T* get() const
    {
        return m_staging_ptr;
    }

    T* staging_data() const
    {
        return m_staging_ptr;
    }

    void* offload_data() const
    {
        return m_offload_ptr;
    }

    size_t size() const
    {
        return m_count;
    }

    T& operator[](size_t idx)
    {
        m_staging_dirty = true;
        return m_staging_ptr[idx];
    }

    const T& operator[](size_t idx) const
    {
        return m_staging_ptr[idx];
    }

    T get(size_t idx) const
    {
        return m_staging_ptr[idx];
    }

    void set(size_t idx, const T& value)
    {
        m_staging_ptr[idx] = value;
        m_staging_dirty = true;
    }

    void copy_to_offload_buffer()
    {
        if (m_bytes == 0 || m_offload_ptr == nullptr || m_staging_ptr == nullptr)
            return;

        m_memory_space->copy_staging_to_offload(m_offload_ptr, m_staging_ptr, m_bytes);
        m_staging_dirty = false;
    }

    void copy_from_offload_buffer()
    {
        if (m_bytes == 0 || m_offload_ptr == nullptr || m_staging_ptr == nullptr)
            return;

        m_memory_space->copy_offload_to_staging(m_staging_ptr, m_offload_ptr, m_bytes);
        m_staging_dirty = false;
    }

    bool needs_offload_sync() const
    {
        return m_staging_dirty;
    }

    void swap(DevicePointer& other) noexcept
    {
        std::swap(m_allocator, other.m_allocator);
        std::swap(m_memory_space, other.m_memory_space);
        std::swap(m_count, other.m_count);
        std::swap(m_bytes, other.m_bytes);
        std::swap(m_alignment, other.m_alignment);
        std::swap(m_staging_ptr, other.m_staging_ptr);
        std::swap(m_offload_ptr, other.m_offload_ptr);
        std::swap(m_owns_staging, other.m_owns_staging);
        std::swap(m_owns_offload, other.m_owns_offload);
        std::swap(m_owns_heap_raw, other.m_owns_heap_raw);
        std::swap(m_staging_dirty, other.m_staging_dirty);
    }

private:
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
        m_staging_dirty = true;
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
        m_staging_dirty = true;
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
        m_staging_dirty = false;
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
    bool m_staging_dirty = false;
};

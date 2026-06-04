#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <parallel.hpp>
#include <Range.hpp>
#include <RandomHelpers.hpp>
#include <IMemorySpace.hpp>

namespace rllm
{
    /** We assume that the ElementType supports default construction, copy assignment, and arithmetic operations.
     * Should be sth like float/float16/int8, not a complex struct.  The X and Y template parameters are the enum
     * types for the row and column indices, respectively, and are only used to determine the matrix dimensions
     * and provide type safety for indexing.
     */
    template <typename ElementType, typename X, typename Y>
    class fixed_size_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        fixed_size_matrix()
            : m_data(ROWS * COLS)
        {}

        fixed_size_matrix(const fixed_size_matrix& other)
            : m_data(ROWS * COLS)
        {
            assign_from(other);
        }

        fixed_size_matrix& operator=(const fixed_size_matrix& other)
        {
            if (this != &other)
                assign_from(other);
            return *this;
        }

        fixed_size_matrix(fixed_size_matrix&& other)
            : m_data(ROWS * COLS)
        {
            assign_from(other);
        }

        fixed_size_matrix& operator=(fixed_size_matrix&& other)
        {
            if (this != &other)
                assign_from(other);
            return *this;
        }

        ~fixed_size_matrix() = default;

        inline void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        inline void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        inline const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        inline ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        template <std::integral XIndex>
        inline ElementType& operator[](XIndex x, Y y)
        {
            return (*this)[static_cast<X>(x), y];
        }

        template <std::integral YIndex>
        inline ElementType& operator[](X x, YIndex y)
        {
            return (*this)[x, static_cast<Y>(y)];
        }

        template <std::integral XIndex, std::integral YIndex>
        inline ElementType& operator[](XIndex x, YIndex y)
        {
            return (*this)[static_cast<X>(x), static_cast<Y>(y)];
        }

        inline const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        template <std::integral XIndex>
        inline const ElementType& operator[](XIndex x, Y y) const
        {
            return (*this)[static_cast<X>(x), y];
        }

        template <std::integral YIndex>
        inline const ElementType& operator[](X x, YIndex y) const
        {
            return (*this)[x, static_cast<Y>(y)];
        }

        template <std::integral XIndex, std::integral YIndex>
        inline const ElementType& operator[](XIndex x, YIndex y) const
        {
            return (*this)[static_cast<X>(x), static_cast<Y>(y)];
        }

        inline void zero()
        {
            m_data.zero();
        }

        inline void fill_rand(ElementType lo, ElementType hi)
        {
            auto* ptr = m_data.get();
            for (size_t i = 0; i < ROWS * COLS; ++i)
                ptr[i] = static_cast<ElementType>(get_random_value(lo, hi));
        }

        ElementType* data()
        {
            return m_data.staging_data();
        }

        const ElementType* data() const
        {
            return m_data.staging_data();
        }

        ElementType* raw_staging_data() const
        {
            return m_data.raw_staging_data();
        }

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        void* raw_offload_data() const
        {
            return m_data.raw_offload_data();
        }
#endif 

#if defined(USE_VULKAN_OFFLOAD)
        VulkanRuntimeBuffer& vulkan_runtime_buffer() const
        {
            return m_data.vulkan_runtime_buffer();
        }
#endif

        DeviceMemoryOwner device_memory_owner() const
        {
            return m_data.device_memory_owner();
        }

        void set_pending_flush(std::function<void()> flush_fn)
        {
            m_data.set_pending_flush(std::move(flush_fn));
        }

        void mark_device_latest()
        {
            m_data.mark_device_latest();
        }

        void copy_to_offload_buffer()
        {
            m_data.copy_to_offload_buffer();
        }

        void copy_row_to_offload_buffer(X x)
        {
            assert(static_cast<size_t>(x) < ROWS);
            m_data.copy_range_to_offload_buffer(static_cast<size_t>(x) * COLS, COLS);
        }

        bool needs_offload_sync() const
        {
            return m_data.needs_offload_sync();
        }

        size_t storage_size_bytes() const
        {
            return ROWS * COLS * sizeof(ElementType);
        }

      private:
                void assign_from(const fixed_size_matrix& other)
                {
                    m_data = other.m_data;
                }

        DevicePointer<ElementType> m_data;
    };
} // namespace rllm

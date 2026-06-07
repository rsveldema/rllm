#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <parallel.hpp>
#include <Range.hpp>
#include <IMemorySpace.hpp>

namespace rllm
{
    /** the number of columns can vary, the number of rows are fixed */
    template <typename ElementType, typename X, typename Y>
    class flexible_cols_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_cols_matrix()
            : m_cols(Y::MAX)
            , m_data(ROWS * COLS)
        {}

        flexible_cols_matrix(Y cols)
            : m_cols(cols)
            , m_data(ROWS * COLS)
        {}

        flexible_cols_matrix(const flexible_cols_matrix& other)
            : m_cols(other.m_cols)
            , m_data(ROWS * COLS)
        {
            m_data = other.m_data;
        }

        flexible_cols_matrix& operator=(const flexible_cols_matrix& other)
        {
            if (this != &other)
            {
                m_data = other.m_data;
                m_cols = other.m_cols;
            }
            return *this;
        }

        flexible_cols_matrix(flexible_cols_matrix&& other)
            : m_cols(other.m_cols)
            , m_data(ROWS * COLS)
        {
            m_data = other.m_data;
        }

        flexible_cols_matrix& operator=(flexible_cols_matrix&& other)
        {
            if (this != &other)
            {
                m_data = other.m_data;
                m_cols = other.m_cols;
            }
            return *this;
        }

        ~flexible_cols_matrix() = default;

        void set_cols(Y cols)
        {
            assert(static_cast<size_t>(cols) <= COLS);
            m_cols = cols;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        void zero()
        {
            m_data.zero();
        }

        const X num_rows() const
        {
            return ROWS;
        }

        Y num_cols() const
        {
            return m_cols;
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
        OffloadMemoryBuffer raw_offload_data() const
        {
            return m_data.raw_offload_data();
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

        bool needs_offload_sync() const
        {
            return m_data.needs_offload_sync();
        }

        size_t storage_size_bytes() const
        {
            return ROWS * COLS * sizeof(ElementType);
        }

      private:
        DevicePointer<ElementType> m_data;
        Y m_cols;
    };

} // namespace rllm

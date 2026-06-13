#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <Range.hpp>
#include <enum_iterator.hpp>
#include <parallel.hpp>


namespace rllm
{
    template <typename T, typename LengthType>
    class fixed_size_obj_vector
    {
      public:
        static constexpr size_t CAPACITY = static_cast<size_t>(LengthType::MAX);

        fixed_size_obj_vector()
        {
            allocate_slots();
        }

        fixed_size_obj_vector(const fixed_size_obj_vector& other)
            : len(other.len)
        {
            allocate_slots();
            copy_slots_from(other);
        }

        fixed_size_obj_vector& operator=(const fixed_size_obj_vector& other)
        {
            if (this != &other)
            {
                ensure_slots();
                len = other.len;
                copy_slots_from(other);
            }
            return *this;
        }

        fixed_size_obj_vector(fixed_size_obj_vector&&) noexcept = default;
        fixed_size_obj_vector& operator=(fixed_size_obj_vector&&) noexcept = default;
        ~fixed_size_obj_vector() = default;

        void set_size(LengthType new_size)
        {
            assert(new_size <= LengthType::MAX);
            ensure_slots();
            len = new_size;
        }

        void push_back(const T& value)
        {
            assert(len < LengthType::MAX);
            *this->m_data[static_cast<size_t>(len)] = value;
            len = static_cast<LengthType>(static_cast<size_t>(len) + 1);
        }

        void push_back(T&& value)
        {
            assert(len < LengthType::MAX);
            *this->m_data[static_cast<size_t>(len)] = std::move(value);
            len = static_cast<LengthType>(static_cast<size_t>(len) + 1);
        }

        const T& back() const
        {
            assert(len > LengthType::START);
            return *this->m_data[static_cast<size_t>(len) - 1];
        }

        void pop_back()
        {
            assert(len > LengthType::START);
            len = static_cast<LengthType>(static_cast<size_t>(len) - 1);
        }

        bool empty() const
        {
            return len == LengthType::START;
        }

        LengthType size() const
        {
            return len;
        }

        auto data()
        {
            return this->m_data.data();
        }

        auto data() const
        {
            return this->m_data.data();
        }

        T& operator[](LengthType index)
        {
            const size_t idx = static_cast<size_t>(index);
            assert(idx < CAPACITY);
            return *this->m_data[idx];
        }

        T& operator[](size_t index)
        {
            assert(index < CAPACITY);
            return *this->m_data[index];
        }

        const T& operator[](LengthType index) const
        {
            const size_t idx = static_cast<size_t>(index);
            assert(idx < CAPACITY);
            return *this->m_data[idx];
        }

        const T& operator[](size_t index) const
        {
            assert(index < CAPACITY);
            return *this->m_data[index];
        }

        void clear()
        {
            len = LengthType::START;
        }

      private:
        void allocate_slots()
        {
            m_data.reserve(CAPACITY);
            for (size_t idx = 0; idx < CAPACITY; ++idx)
                m_data.push_back(std::make_unique<T>());
        }

        void ensure_slots()
        {
            if (m_data.size() == CAPACITY)
                return;
            m_data.clear();
            allocate_slots();
        }

        void copy_slots_from(const fixed_size_obj_vector& other)
        {
            assert(m_data.size() == CAPACITY);
            assert(other.m_data.size() == CAPACITY);
            for (size_t idx = 0; idx < CAPACITY; ++idx)
                *m_data[idx] = *other.m_data[idx];
        }

        std::vector<std::unique_ptr<T>> m_data;
        LengthType len = LengthType::START;
    };

} // namespace rllm

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>

#include <Range.hpp>
#include <enum_iterator1D.hpp>
#include <cpu/cpu_data.hpp>

namespace rllm
{
    template <typename T, typename LengthType>
    class cpu_fixed_vector : public cpu_data<T>
    {
      public:
        static constexpr size_t CAPACITY = static_cast<size_t>(LengthType::MAX);

        cpu_fixed_vector()
            : cpu_data<T>(CAPACITY)
        {}

        cpu_fixed_vector(const cpu_fixed_vector& other)
            : cpu_data<T>(CAPACITY)
            , len(other.len)
        {
            std::copy(other.data(), other.data() + CAPACITY, this->data());
        }

        cpu_fixed_vector& operator=(const cpu_fixed_vector& other)
        {
            if (this != &other)
            {
                std::copy(other.data(), other.data() + CAPACITY, this->data());
                len = other.len;
            }
            return *this;
        }

        cpu_fixed_vector(cpu_fixed_vector&&) noexcept = default;
        cpu_fixed_vector& operator=(cpu_fixed_vector&&) noexcept = default;
        ~cpu_fixed_vector() = default;

        void set_size(LengthType new_size)
        {
            assert(new_size <= LengthType::MAX);
            len = new_size;
        }

        void sub_array(cpu_fixed_vector& result, LengthType length) const
        {
            assert(length <= len);
            const auto* data = this->data();
            result.clear();
            for (const auto i : enum_iterator1D<LengthType>(length))
            {
                const auto tok = data[static_cast<size_t>(i)];
                result.push_back(tok);
            }
            result.len = length;
        }

        void push_back(T value)
        {
            assert(len < LengthType::MAX);
            this->data()[static_cast<size_t>(len)] = value;
            len = static_cast<LengthType>(static_cast<size_t>(len) + 1);
        }

        const T& back() const
        {
            assert(len > LengthType::START);
            return this->data()[static_cast<size_t>(len) - 1];
        }

        void pop_back()
        {
            assert(len > LengthType::START);
            len = static_cast<LengthType>(static_cast<size_t>(len) - 1);
        }

        bool empty() const { return len == LengthType::START; }
        LengthType size() const { return len; }
        size_t storage_size_bytes() const { return static_cast<size_t>(len) * sizeof(T); }

        T& operator[](LengthType index)
        {
            assert(index < LengthType::MAX);
            return this->data()[static_cast<size_t>(index)];
        }

        T& operator[](size_t index)
        {
            assert(index < static_cast<size_t>(len));
            return this->data()[index];
        }

        const T& operator[](LengthType index) const
        {
            assert(index < LengthType::MAX);
            return this->data()[static_cast<size_t>(index)];
        }

        const T& operator[](size_t index) const
        {
            assert(index < static_cast<size_t>(len));
            return this->data()[index];
        }

        void clear() { len = LengthType::START; }

      private:
        LengthType len = LengthType::START;
    };
} // namespace rllm

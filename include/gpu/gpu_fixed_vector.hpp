#pragma once

#include <cassert>
#include <cstddef>

#include <IMemorySpace.hpp>
#include <offloadable_data.hpp>
#include <cpu/cpu_fixed_vector.hpp>
#include <rllm_vulkan_runtime.hpp>

namespace rllm
{
    template <typename T, typename LengthType>
    class gpu_fixed_vector : public offloadable_data<T>
    {
      public:
        static constexpr size_t CAPACITY = static_cast<size_t>(LengthType::MAX);

        gpu_fixed_vector()
            : offloadable_data<T>(CAPACITY)
        {}

        /** D2D copy constructor — copies device buffer contents via Vulkan. */
        gpu_fixed_vector(const gpu_fixed_vector& other)
            : offloadable_data<T>(CAPACITY)
            , len(other.len)
        {
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(len) * sizeof(T));
            if (bytes > 0)
                this->m_data.device_buffer().copy_from(
                    rllm::vulkan_runtime::get_queue(0),
                    other.m_data.device_buffer(),
                    bytes);
        }

        /** D2D copy assignment — copies device buffer contents via Vulkan. */
        gpu_fixed_vector& operator=(const gpu_fixed_vector& other)
        {
            if (this != &other)
            {
                len = other.len;
                const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(len) * sizeof(T));
                if (bytes > 0)
                    this->m_data.device_buffer().copy_from(
                        rllm::vulkan_runtime::get_queue(0),
                        other.m_data.device_buffer(),
                        bytes);
            }
            return *this;
        }

        gpu_fixed_vector(gpu_fixed_vector&&) = default;
        gpu_fixed_vector& operator=(gpu_fixed_vector&&) = default;

        ~gpu_fixed_vector() = default;

        void set_size(LengthType new_size)
        {
            assert(new_size <= LengthType::MAX);
            len = new_size;
        }

        bool empty() const { return len == LengthType::START; }
        LengthType size() const { return len; }
        size_t storage_size_bytes() const { return static_cast<size_t>(len) * sizeof(T); }

        void clear() { len = LengthType::START; }

        /** H2D: upload from a cpu_fixed_vector. Also updates size to match src. */
        void copy_from_cpu(const cpu_fixed_vector<T, LengthType>& src)
        {
            len = src.size();
            const auto n = static_cast<size_t>(len);
            if (n > 0)
                this->m_data.copy_range_to_offload_buffer(
                    const_cast<VBaseHostBuffer&>(src.vk_host_buffer()), 0, n);
        }

        /** D2H: download into a cpu_fixed_vector. Also updates dst size to match. */
        void copy_to_cpu(cpu_fixed_vector<T, LengthType>& dst) const
        {
            dst.set_size(len);
            if (static_cast<size_t>(len) > 0)
                this->m_data.copy_from_offload_buffer(dst.vk_host_buffer());
        }

      private:
        LengthType len = LengthType::START;
    };
} // namespace rllm

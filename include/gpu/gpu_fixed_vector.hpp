#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>

#include <logging.hpp>

#include <IMemorySpace.hpp>
#include <cpu/cpu_fixed_vector.hpp>
#include <offloadable_data.hpp>
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

        gpu_fixed_vector(const gpu_fixed_vector& other) = delete;
        gpu_fixed_vector& operator=(const gpu_fixed_vector& other) = delete;
        gpu_fixed_vector(gpu_fixed_vector&&) = default;
        gpu_fixed_vector& operator=(gpu_fixed_vector&&) = default;

        ~gpu_fixed_vector() = default;

        void set_size(LengthType new_size)
        {
            assert(new_size <= LengthType::MAX);
            len = new_size;
        }

        bool empty() const
        {
            return len == LengthType::START;
        }
        LengthType size() const
        {
            return len;
        }
        size_t storage_size_bytes() const
        {
            return static_cast<size_t>(len) * sizeof(T);
        }

        void clear()
        {
            len = LengthType::START;
        }

        /** H2D: upload from a cpu_fixed_vector. Also updates size to match src. */
        void copy_from_cpu(VulkanQueue& queue, const cpu_fixed_vector<T, LengthType>& src)
        {
            len = src.size();
            const auto n = static_cast<size_t>(len);
            if (n == 0)
            {
                LOG_ERROR("tried to copy zer0 bytes");
                abort();
            }
            
            this->m_data.copy_range_to_offload_buffer(queue, const_cast<VBaseHostBuffer&>(src.vk_host_buffer()), 0, n);
        }

        /** D2H: download into a cpu_fixed_vector. Also updates dst size to match. */
        void copy_to_cpu(VulkanQueue& queue, cpu_fixed_vector<T, LengthType>& dst) const
        {
            dst.set_size(len);
            if (static_cast<size_t>(len) == 0)
            {
                LOG_ERROR("tried to copy zer0 bytes");
                abort();
            }
             
            this->m_data.copy_from_offload_buffer(queue, dst.vk_host_buffer());
        }

      private:
        LengthType len = LengthType::START;
    };
} // namespace rllm

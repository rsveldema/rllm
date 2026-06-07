#pragma once

#include <functional>
#include <IMemorySpace.hpp>
#include <device_pointer.hpp>

namespace rllm
{
    template <typename T>
    class offloadable_data
    {
    public:
        offloadable_data(size_t elts)
        : m_data(elts)
        {}

        void zero()
        {
            m_data.zero();
        }

        T* data()
        {
            return m_data.get();
        }

        const T* data() const
        {
            return m_data.get();
        }

        T* raw_staging_data() const
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

        void copy_to_offload_buffer(std::string_view site = {}, std::string_view parameter = {})
        {
            m_data.copy_to_offload_buffer(site, parameter);
        }

        bool needs_offload_sync() const
        {
            return m_data.needs_offload_sync();
        }
        
      protected:
        DevicePointer<T> m_data;
    };
} // namespace rllm
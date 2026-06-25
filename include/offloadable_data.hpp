#pragma once

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

        VBaseDeviceBuffer& device_buffer() const
        {
            return m_data.device_buffer();
        }

        void mark_device_latest()
        {
            m_data.mark_device_latest();
        }

        /** H2D: upload from an external host-visible buffer.
         *  Called explicitly from copy_from_cpu() on the concrete gpu_* types. */
        void copy_to_offload_buffer(VBaseHostBuffer& src,
                                    std::string_view site = {},
                                    std::string_view parameter = {})
        {
            m_data.copy_to_offload_buffer(src, site, parameter);
        }

        /** No-op kept for generated-code compatibility.
         *  The actual upload happens through copy_from_cpu(cpu_T&). */
        void copy_to_offload_buffer(std::string_view = {}, std::string_view = {}) {}

        /** D2H: download to an external host-visible buffer.
         *  Called explicitly from copy_to_cpu() on the concrete gpu_* types. */
        void copy_from_offload_buffer(VBaseHostBuffer& dst)
        {
            m_data.copy_from_offload_buffer(dst);
        }

        /** No-op kept for compatibility; use copy_to_cpu(cpu_T&) for D2H. */
        void sync_from_device() {}

      protected:
        DevicePointer<T> m_data;
    };
} // namespace rllm


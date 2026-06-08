#pragma once

#include <cstdlib>
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <vector>

#include <IMemorySpace.hpp>
#include <logging.hpp>

#if defined(RLLM_ENABLE_STATISTICS)
#include <parallel.hpp>
#endif

class HipMemorySpace final : public IMemorySpace
{
public:
    explicit HipMemorySpace()
    {
        const char *tst = getenv("HSA_XNACK");
        assert(tst != nullptr);
        assert(std::string(tst) == "1");
    }

    ~HipMemorySpace() override
    {
    }

    void copy_staging_to_offload(const OffloadMemoryBuffer& offload_dst, size_t dst_offset, const OnHostStagingBuffer& staging_src, size_t src_offset, size_t bytes, std::string_view site = {}, std::string_view parameter = {}) override
    {
        assert(bytes != 0);
#if defined(RLLM_ENABLE_STATISTICS)
        {
            std::string_view effective_site = site;
            if (effective_site.empty())
                effective_site = parallel::g_vulkan_dispatch_params.site;
            std::string_view effective_parameter = parameter;
            if (effective_site.empty())
                effective_parameter = parallel::g_vulkan_dispatch_params.parameter;
            if (!effective_site.empty())
                parallel::statistics.record_host_to_device_buffer_copy(effective_site, effective_parameter, bytes);
        }
#endif
        const hipError_t status = hipMemcpy(offload_dst.get() + dst_offset, staging_src.get() + src_offset, bytes, hipMemcpyHostToDevice);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload upload failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

    void copy_offload_to_staging(const OnHostStagingBuffer& staging_dst, size_t dst_offset, const OffloadMemoryBuffer& offload_src, size_t src_offset, size_t bytes, std::string_view site = {}, std::string_view parameter = {}) override
    {
        assert(bytes != 0);
#if defined(RLLM_ENABLE_STATISTICS)
        {
            std::string_view effective_site = site;
            if (effective_site.empty())
                effective_site = parallel::g_vulkan_dispatch_params.site;
            std::string_view effective_parameter = parameter;
            if (effective_site.empty())
                effective_parameter = parallel::g_vulkan_dispatch_params.parameter;
            if (!effective_site.empty())
                parallel::statistics.record_device_to_host_buffer_copy(effective_site, effective_parameter, bytes);
        }
#endif
        const hipError_t status = hipMemcpy(staging_dst.get() + dst_offset, offload_src.get() + src_offset, bytes, hipMemcpyDeviceToHost);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload download failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }
    void copy_offload_to_offload(const OffloadMemoryBuffer& offload_dst, size_t dst_offset, const OffloadMemoryBuffer& offload_src, size_t src_offset, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemcpy(offload_dst.get() + dst_offset, offload_src.get() + src_offset, bytes, hipMemcpyDeviceToDevice);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload device copy failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

    void zero_offload(const OffloadMemoryBuffer& offload_dst, size_t offset, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemset(offload_dst.get() + offset, 0, bytes);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload memset failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

    OnHostStagingBuffer allocate_staging(size_t bytes) override
    {
        void *a = hipHostMalloc(bytes);
        return OnHostStagingBuffer { a };
    }

    void release_staging(OnHostStagingBuffer& ref) override
    {
        hipHostFree(ref.get());
        ref.invalidate();
    }


    OffloadMemoryBuffer allocate_offload(size_t bytes) override {
        void *a = hipMalloc(bytes);
        return OnHostStagingBuffer { a };
    }

    void release_offload(OffloadMemoryBuffer& ref) override
    {
        hipFree(ref.get());
        ref.invalidate();
    }
};
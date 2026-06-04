#pragma once

#include <cstdlib>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <vector>

#include <IMemorySpace.hpp>
#include <logging.hpp>

class HipMemorySpace final : public IMemorySpace
{
public:
    static constexpr size_t DEFAULT_POOL_BYTES = 8ULL * 1024ULL * 1024ULL * 1024ULL;

    explicit HipMemorySpace(size_t pool_bytes = DEFAULT_POOL_BYTES)
        : staging_storage_(pool_bytes)
        , offload_storage_(nullptr)
    {
        const hipError_t status = hipMalloc(&offload_storage_, pool_bytes);
        if (status != hipSuccess)
        {
            LOG_ERROR("HipMemorySpace: hipMalloc failed for offload pool: {}", hipGetErrorString(status));
            throw std::runtime_error("HipMemorySpace: hipMalloc failed for offload pool");
        }
    }

    ~HipMemorySpace() override
    {
        if (offload_storage_ != nullptr)
        {
            const hipError_t status = hipFree(offload_storage_);
            if (status != hipSuccess)
                LOG_ERROR("HipMemorySpace: hipFree failed for offload pool: {}", hipGetErrorString(status));
        }
    }

    size_t get_total_size() const override
    {
        return staging_storage_.size();
    }

    void* getMemory() override
    {
        return staging_storage_.data();
    }

    void* get_offload_memory() override
    {
        return offload_storage_;
    }

    void copy_staging_to_offload(void* offload_dst, const void* staging_src, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemcpy(offload_dst, staging_src, bytes, hipMemcpyDefault);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload upload failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

    void copy_offload_to_staging(void* staging_dst, const void* offload_src, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemcpy(staging_dst, offload_src, bytes, hipMemcpyDefault);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload download failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

    void copy_offload_to_offload(void* offload_dst, const void* offload_src, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemcpy(offload_dst, offload_src, bytes, hipMemcpyDefault);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload device copy failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

    void zero_offload(void* offload_dst, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemset(offload_dst, 0, bytes);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload memset failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

private:
    std::vector<std::byte> staging_storage_;
    void* offload_storage_;
};
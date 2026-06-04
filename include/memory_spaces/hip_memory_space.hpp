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

class HipMemorySpace final : public IMemorySpace
{
public:
    static constexpr size_t DEFAULT_POOL_BYTES = 8ULL * 1024ULL * 1024ULL * 1024ULL;

    explicit HipMemorySpace(size_t pool_bytes = DEFAULT_POOL_BYTES)
        : staging_size_(pool_bytes / 2), offload_size_(pool_bytes / 2),
          staging_offset_(0), offload_offset_(0)
    {
        staging_storage_.resize(pool_bytes / 2);
        // HIP uses hipMalloc for device memory
    }

    ~HipMemorySpace() override
    {
        // Free all individual offload allocations
        for (void* ptr : individual_offload_ptrs_)
        {
            hipFree(ptr);
        }
    }

    size_t get_total_size() const override
    {
        return staging_storage_.size() + offload_size_;
    }

    void* getMemory() override
    {
        return staging_storage_.data();
    }

    void* get_offload_memory() override
    {
        // For HIP, device memory is allocated individually via hipMalloc
        return nullptr; // No contiguous pool for device
    }

    void copy_staging_to_offload(void* offload_dst, const void* staging_src, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemcpy(offload_dst, staging_src, bytes, hipMemcpyHostToDevice);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload upload failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

    void copy_offload_to_staging(void* staging_dst, const void* offload_src, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemcpy(staging_dst, offload_src, bytes, hipMemcpyDeviceToHost);
        if (status != hipSuccess)
        {
            LOG_ERROR("Fatal: HIP offload download failed: {}", hipGetErrorString(status));
            std::abort();
        }
    }

    void copy_offload_to_offload(void* offload_dst, const void* offload_src, size_t bytes) override
    {
        assert(bytes != 0);
        const hipError_t status = hipMemcpy(offload_dst, offload_src, bytes, hipMemcpyDeviceToDevice);
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

    void* allocate_staging(size_t bytes) override
    {
        size_t aligned = (bytes + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
        if (staging_offset_ + aligned > staging_size_)
        {
            return nullptr;
        }
        void* ptr = static_cast<std::byte*>(staging_storage_.data()) + staging_offset_;
        staging_offset_ += aligned;
        return ptr;
    }

    void* allocate_offload(size_t bytes) override
    {
        hipError_t status = hipMalloc(&offload_ptr_, bytes);
        if (status != hipSuccess)
        {
            LOG_ERROR("HIP offload allocation failed: {}", hipGetErrorString(status));
            return nullptr;
        }
        individual_offload_ptrs_.push_back(offload_ptr_);
        return offload_ptr_;
    }

    void release_staging(void* /*ptr*/, size_t /*bytes*/) override
    {
        // No individual deallocation in bump allocator; reset handles bulk release
    }

    void release_offload(void* ptr, size_t /*bytes*/) override
    {
        for (auto it = individual_offload_ptrs_.begin(); it != individual_offload_ptrs_.end(); ++it)
        {
            if (*it == ptr)
            {
                hipFree(*it);
                individual_offload_ptrs_.erase(it);
                return;
            }
        }
    }

    void reset() override
    {
        staging_offset_ = 0;
        // Free all offloaded allocations
        for (void* ptr : individual_offload_ptrs_)
        {
            hipFree(ptr);
        }
        individual_offload_ptrs_.clear();
    }

private:
    std::vector<std::byte> staging_storage_;
    void* offload_ptr_ = nullptr;
    size_t staging_size_ = 0;
    size_t offload_size_ = 0;
    size_t staging_offset_ = 0;
    std::vector<void*> individual_offload_ptrs_;
};
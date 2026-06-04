#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <IMemorySpace.hpp>

class HostMemorySpace final : public IMemorySpace
{
public:
    static constexpr size_t DEFAULT_POOL_BYTES = 256ULL * 1024ULL * 1024ULL;

    explicit HostMemorySpace(size_t pool_bytes = DEFAULT_POOL_BYTES)
        : storage_(pool_bytes)
        , staging_base_(storage_.data())
#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        , offload_base_(storage_.data() + (pool_bytes / 2))
        , staging_size_(pool_bytes / 2)
        , offload_size_(pool_bytes / 2)
#else
        , offload_base_(storage_.data())
        , staging_size_(pool_bytes)
        , offload_size_(0)
#endif
        , staging_offset_(0)
        , offload_offset_(0)
    {}

    size_t get_total_size() const override
    {
        return storage_.size();
    }

    void* getMemory() override
    {
        return storage_.data();
    }

    void* get_offload_memory() override
    {
        return static_cast<std::byte*>(storage_.data()) + staging_size_;
    }

    void copy_staging_to_offload(void* offload_dst, const void* staging_src, size_t bytes) override
    {
        if (bytes == 0)
            return;
        std::memcpy(offload_dst, staging_src, bytes);
    }

    void copy_offload_to_staging(void* staging_dst, const void* offload_src, size_t bytes) override
    {
        if (bytes == 0)
            return;
        std::memcpy(staging_dst, offload_src, bytes);
    }

    void copy_offload_to_offload(void* offload_dst, const void* offload_src, size_t bytes) override
    {
        if (bytes == 0)
            return;
        std::memcpy(offload_dst, offload_src, bytes);
    }

    void zero_offload(void* offload_dst, size_t bytes) override
    {
        if (bytes == 0)
            return;
        std::memset(offload_dst, 0, bytes);
    }

    void* allocate_staging(size_t bytes) override
    {
        size_t aligned = (bytes + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
        if (staging_offset_ + aligned > staging_size_)
        {
            return nullptr;
        }
        void* ptr = static_cast<std::byte*>(staging_base_) + staging_offset_;
        staging_offset_ += aligned;
        return ptr;
    }

    void* allocate_offload(size_t bytes) override
    {
        size_t aligned = (bytes + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
        if (offload_offset_ + aligned > offload_size_)
        {
            return nullptr;
        }
        void* ptr = static_cast<std::byte*>(offload_base_) + offload_offset_;
        offload_offset_ += aligned;
        return ptr;
    }

    void release_staging(void* /*ptr*/, size_t /*bytes*/) override
    {
        // No individual deallocation in bump allocator; reset handles bulk release
    }

    void release_offload(void* /*ptr*/, size_t /*bytes*/) override
    {
        // No individual deallocation in bump allocator; reset handles bulk release
    }

    void reset() override
    {
        staging_offset_ = 0;
        offload_offset_ = 0;
    }

private:
    std::vector<std::byte> storage_;
    std::byte* staging_base_ = nullptr;
    std::byte* offload_base_ = nullptr;
    size_t staging_size_ = 0;
    size_t offload_size_ = 0;
    size_t staging_offset_ = 0;
    size_t offload_offset_ = 0;
};

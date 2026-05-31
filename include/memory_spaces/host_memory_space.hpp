#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

#include <IMemorySpace.hpp>

class HostMemorySpace final : public IMemorySpace
{
public:
    static constexpr size_t DEFAULT_POOL_BYTES = 256ULL * 1024ULL * 1024ULL;

    explicit HostMemorySpace(size_t pool_bytes = DEFAULT_POOL_BYTES)
        : storage_(pool_bytes)
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
        return storage_.data();
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

private:
    std::vector<std::byte> storage_;
};

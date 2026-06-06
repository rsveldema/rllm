#pragma once

#include <cstdlib>
#include <cassert>

#include <IMemorySpace.hpp>

class HostMemorySpace final : public IMemorySpace
{
public:
    explicit HostMemorySpace()
    {}

    OnHostStagingBuffer allocate_staging(size_t bytes)
    { 
        return OnHostStagingBuffer{ malloc(bytes) };
    }

    OffloadMemoryBuffer allocate_offload(size_t bytes)
    {
        assert(false);
        abort();        
    }

    void copy_staging_to_offload(const OffloadMemoryBuffer& offload_dst, const OnHostStagingBuffer& staging_src, size_t bytes)
    {
        assert(false);
        abort();
    }

    void copy_offload_to_staging(const OnHostStagingBuffer& staging_dst, const OffloadMemoryBuffer& offload_src, size_t bytes)
    {
        assert(false);
        abort();
    }

    void copy_offload_to_offload(const OffloadMemoryBuffer& offload_dst, const OffloadMemoryBuffer& offload_src, size_t bytes)
    {
        assert(false);
        abort();
    }

    void zero_offload(const OffloadMemoryBuffer& offload_dst, size_t bytes)
    {
        assert(false);
        abort();
    }
    
    void release_staging(OnHostStagingBuffer& ref)
    {
        free(ref.get());
        ref.invalidate();
    }

    void release_offload(OffloadMemoryBuffer& ref) 
    {
        assert(false);
        abort();
    }
};

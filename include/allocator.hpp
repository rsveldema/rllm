#pragma once

#include <cstddef>
#include <new>
#include <stdexcept>

#include <IMemorySpace.hpp>

class Allocator
{
public:
    struct Region
    {
        void* base;
        size_t size;
        size_t offset;
    };

    explicit Allocator(IMemorySpace& memory_space)
        : memory_space_(memory_space)
        , staging_region_{memory_space.getMemory(), memory_space.get_total_size(), 0}
        , offload_region_{memory_space.get_offload_memory(), memory_space.get_total_size(), 0}
    {
        if (staging_region_.base == nullptr || offload_region_.base == nullptr || staging_region_.size == 0)
            throw std::runtime_error("Allocator requires a non-empty memory space");
    }

    /** allocate a block of memory from the staging region */
    void* allocate(size_t size)
    {
        return allocate_staging(size, alignof(std::max_align_t));
    }

    void* allocate_staging(size_t size, size_t alignment)
    {
        if (size == 0)
            return staging_region_.base;

        return allocate_from_region(staging_region_, size, alignment);
    }

    void* allocate_offload(size_t size, size_t alignment)
    {
        if (size == 0)
            return offload_region_.base;

        return allocate_from_region(offload_region_, size, alignment);
    }

    void release_staging([[maybe_unused]] void* ptr, [[maybe_unused]] size_t alignment)
    {
        // Bump allocator: staging allocations are released by reset()/allocator lifetime.
    }

    void release_offload([[maybe_unused]] void* ptr, [[maybe_unused]] size_t alignment)
    {
        // Bump allocator: offload allocations are released by reset()/allocator lifetime.
    }

    IMemorySpace& memory_space()
    {
        return memory_space_;
    }

    void reset()
    {
        staging_region_.offset = 0;
        offload_region_.offset = 0;
    }

private:
    void* allocate_from_region(Region& region, size_t size, size_t alignment)
    {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            throw std::invalid_argument("Allocator alignment must be a power of two");

        const size_t alignment_mask = alignment - 1;

        if (size > (static_cast<size_t>(-1) - alignment_mask))
            throw std::bad_alloc();

        const size_t aligned_size = (size + alignment_mask) & ~alignment_mask;
        if (aligned_size > (region.size - region.offset))
            throw std::bad_alloc();

        auto* const base = static_cast<std::byte*>(region.base);
        void* const out = base + region.offset;
        region.offset += aligned_size;
        return out;
    }

    IMemorySpace& memory_space_;
    Region staging_region_;
    Region offload_region_;
};

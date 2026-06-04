#pragma once

#include <cstddef>
#include <cstdint>
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
        release_from_region(staging_region_, ptr);
    }

    void release_offload([[maybe_unused]] void* ptr, [[maybe_unused]] size_t alignment)
    {
        release_from_region(offload_region_, ptr);
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
    struct AllocationHeader
    {
        uint64_t magic;
        size_t start_offset;
        size_t end_offset;
    };

    static constexpr uint64_t ALLOCATION_MAGIC = 0x524C4C4D414C4C4FULL; // RLLMALLO

    static size_t align_up(size_t value, size_t alignment)
    {
        const size_t alignment_mask = alignment - 1;
        if (value > (static_cast<size_t>(-1) - alignment_mask))
            throw std::bad_alloc();
        return (value + alignment_mask) & ~alignment_mask;
    }

    void* allocate_from_region(Region& region, size_t size, size_t alignment)
    {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            throw std::invalid_argument("Allocator alignment must be a power of two");

        const size_t start_offset = region.offset;
        const size_t header_end_offset = start_offset + sizeof(AllocationHeader);
        if (header_end_offset < start_offset)
            throw std::bad_alloc();

        const size_t user_offset = align_up(header_end_offset, alignment);
        if (size > (static_cast<size_t>(-1) - user_offset))
            throw std::bad_alloc();

        const size_t end_offset = user_offset + size;
        if (end_offset > region.size)
            throw std::bad_alloc();

        auto* const base = static_cast<std::byte*>(region.base);
        auto* const header = reinterpret_cast<AllocationHeader*>(base + user_offset - sizeof(AllocationHeader));
        header->magic = ALLOCATION_MAGIC;
        header->start_offset = start_offset;
        header->end_offset = end_offset;

        void* const out = base + user_offset;
        region.offset = end_offset;
        return out;
    }

    void release_from_region(Region& region, void* ptr)
    {
        if (ptr == nullptr || region.base == nullptr)
            return;

        auto* const base = static_cast<std::byte*>(region.base);
        auto* const byte_ptr = static_cast<std::byte*>(ptr);
        if (byte_ptr < base + sizeof(AllocationHeader) || byte_ptr > base + region.size)
            return;

        auto* const header = reinterpret_cast<AllocationHeader*>(byte_ptr - sizeof(AllocationHeader));
        if (header->magic != ALLOCATION_MAGIC || header->end_offset != region.offset)
            return;

        region.offset = header->start_offset;
        header->magic = 0;
    }

    IMemorySpace& memory_space_;
    Region staging_region_;
    Region offload_region_;
};

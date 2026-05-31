#pragma once

#include <IMemorySpace.hpp>

// Backend selection is controlled via cmake:
//   -DPARALLEL_BACKEND=openmp|fastfork|sequential
// CMake maps that to one of:
//   USE_OPENMP, USE_FASTFORK, or neither (sequential fallback).

#if defined(USE_VULKAN_OFFLOAD)
#include <memory_spaces/vulkan_memory_space.hpp>
#elif defined(USE_HIP_OFFLOAD)
#include <memory_spaces/hip_memory_space.hpp>
#else
#include <memory_spaces/host_memory_space.hpp>
#endif
#include <allocator.hpp>

inline IMemorySpace* IMemorySpace::get_instance()
{
#if defined(USE_VULKAN_OFFLOAD)
    static VulkanMemorySpace instance;
#elif defined(USE_HIP_OFFLOAD)
    static HipMemorySpace instance;
#else
    static HostMemorySpace instance;
#endif
    return &instance;
}

inline Allocator& get_offload_allocator()
{
    static thread_local Allocator allocator(*IMemorySpace::get_instance());
    return allocator;
}

#include <device_pointer.hpp>


#if defined(USE_OPENMP)
#include <parallel_openmp.hpp>
#elif defined(USE_FASTFORK)
#include <parallel_fastfork.hpp>
#else
#include <parallel_sequential.hpp>
#endif

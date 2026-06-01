#pragma once

#include <atomic>
#include <print>

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

namespace parallel {
    const char* backend_name();
    void print_vulkan_provider();

    class Statistics
    {
      public:
        void record_host_to_device_buffer_copy()
        {
            m_host_to_device_buffer_copies.fetch_add(1, std::memory_order_relaxed);
        }

        void record_device_to_host_buffer_copy()
        {
            m_device_to_host_buffer_copies.fetch_add(1, std::memory_order_relaxed);
        }

        size_t host_to_device_buffer_copies() const
        {
            return m_host_to_device_buffer_copies.load(std::memory_order_relaxed);
        }

        size_t device_to_host_buffer_copies() const
        {
            return m_device_to_host_buffer_copies.load(std::memory_order_relaxed);
        }

        void reset_buffer_copy_counters()
        {
            m_host_to_device_buffer_copies.store(0, std::memory_order_relaxed);
            m_device_to_host_buffer_copies.store(0, std::memory_order_relaxed);
        }

        void print_statistics() const
        {
            std::println("Parallel statistics:");
            std::println("  Host-to-device buffer copies: {}", host_to_device_buffer_copies());
            std::println("  Device-to-host buffer copies: {}", device_to_host_buffer_copies());
        }

      private:
        std::atomic<size_t> m_host_to_device_buffer_copies{0};
        std::atomic<size_t> m_device_to_host_buffer_copies{0};
    };

    extern Statistics statistics;
}

#include <device_pointer.hpp>


#if defined(USE_OPENMP)
#include <parallel_openmp.hpp>
#elif defined(USE_FASTFORK)
#include <parallel_fastfork.hpp>
#else
#include <parallel_sequential.hpp>
#endif

#define RLLM_STRINGIZE_IMPL(x) #x
#define RLLM_STRINGIZE(x) RLLM_STRINGIZE_IMPL(x)
#define RLLM_DO_PRAGMA(x) _Pragma(RLLM_STRINGIZE(x))

#if defined(USE_OPENMP) && !defined(USE_HIP_OFFLOAD) && !defined(USE_VULKAN_OFFLOAD)
#define RLLM_OMP_SIMD RLLM_DO_PRAGMA(omp simd)
#define RLLM_OMP_SIMD_REDUCTION_PLUS(var) RLLM_DO_PRAGMA(omp simd reduction(+:var))
#else
#define RLLM_OMP_SIMD
#define RLLM_OMP_SIMD_REDUCTION_PLUS(var)
#endif

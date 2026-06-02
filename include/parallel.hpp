#pragma once

#include <algorithm>
#include <atomic>
#include <mutex>
#include <print>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

#if !defined(USE_VULKAN_OFFLOAD)
namespace parallel::detail
{
    inline void cpu_barrier()
    {
    }

    inline void cpu_atomic_add(float& dst, const float value)
    {
        std::atomic_ref<float> atomic_dst(dst);
        float current = atomic_dst.load(std::memory_order_relaxed);
        while (!atomic_dst.compare_exchange_weak(
            current,
            current + value,
            std::memory_order_relaxed,
            std::memory_order_relaxed
        ))
        {
        }
    }

    inline void cpu_atomic_max(float& dst, const float value)
    {
        std::atomic_ref<float> atomic_dst(dst);
        float current = atomic_dst.load(std::memory_order_relaxed);
        while (current < value && !atomic_dst.compare_exchange_weak(
            current,
            value,
            std::memory_order_relaxed,
            std::memory_order_relaxed
        ))
        {
        }
    }
}

#define barrier() ::parallel::detail::cpu_barrier()
#define atomicAdd(dst, value) ::parallel::detail::cpu_atomic_add((dst), (value))
#define atomicMax(dst, value) ::parallel::detail::cpu_atomic_max((dst), (value))
#endif

namespace parallel {
    const char* backend_name();
    void print_vulkan_provider();

    class Statistics
    {
      public:
        struct CopySiteBreakdown
        {
            std::string site;
            size_t host_to_device = 0;
            size_t device_to_host = 0;
        };

        struct CopyParameterBreakdown
        {
            std::string site;
            std::string parameter;
            size_t host_to_device = 0;
            size_t device_to_host = 0;
        };

        void record_host_to_device_buffer_copy(std::string_view site = {}, std::string_view parameter = {})
        {
            m_host_to_device_buffer_copies.fetch_add(1, std::memory_order_relaxed);
            record_copy_site(site, true);
            record_copy_parameter(site, parameter, true);
        }

        void record_device_to_host_buffer_copy(std::string_view site = {}, std::string_view parameter = {})
        {
            m_device_to_host_buffer_copies.fetch_add(1, std::memory_order_relaxed);
            record_copy_site(site, false);
            record_copy_parameter(site, parameter, false);
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

            std::lock_guard<std::mutex> lock(m_copy_site_mutex);
            m_copy_sites.clear();
            m_copy_parameters.clear();
        }

        void print_statistics() const
        {
            std::println("Parallel statistics:");
            std::println("  Host-to-device buffer copies: {}", host_to_device_buffer_copies());
            std::println("  Device-to-host buffer copies: {}", device_to_host_buffer_copies());

            const auto top_sites = top_copy_sites();
            if (!top_sites.empty())
            {
                std::println("  Top copy sites:");
                for (const auto& site : top_sites)
                {
                    std::println(
                        "    {}: H2D={}, D2H={}",
                        site.site,
                        site.host_to_device,
                        site.device_to_host
                    );
                }
            }

            const auto top_parameters = top_copy_parameters();
            if (!top_parameters.empty())
            {
                std::println("  Top copy parameters:");
                for (const auto& item : top_parameters)
                {
                    std::println(
                        "    {} [{}]: H2D={}, D2H={}",
                        item.site,
                        item.parameter,
                        item.host_to_device,
                        item.device_to_host
                    );
                }
            }
        }

        std::vector<CopySiteBreakdown> top_copy_sites(size_t limit = 10) const
        {
            std::lock_guard<std::mutex> lock(m_copy_site_mutex);

            std::vector<CopySiteBreakdown> sites;
            sites.reserve(m_copy_sites.size());
            for (const auto& [site, counts] : m_copy_sites)
            {
                sites.push_back(CopySiteBreakdown{
                    .site = site,
                    .host_to_device = counts.host_to_device,
                    .device_to_host = counts.device_to_host,
                });
            }

            std::sort(sites.begin(), sites.end(), [](const auto& lhs, const auto& rhs) {
                const size_t lhs_total = lhs.host_to_device + lhs.device_to_host;
                const size_t rhs_total = rhs.host_to_device + rhs.device_to_host;
                if (lhs_total != rhs_total)
                    return lhs_total > rhs_total;
                return lhs.site < rhs.site;
            });

            if (sites.size() > limit)
                sites.resize(limit);

            return sites;
        }

        std::vector<CopyParameterBreakdown> top_copy_parameters(size_t limit = 10) const
        {
            std::lock_guard<std::mutex> lock(m_copy_site_mutex);

            std::vector<CopyParameterBreakdown> items;
            items.reserve(m_copy_parameters.size());
            for (const auto& [key, counts] : m_copy_parameters)
            {
                const size_t split = key.find('\n');
                if (split == std::string::npos)
                    continue;

                items.push_back(CopyParameterBreakdown{
                    .site = key.substr(0, split),
                    .parameter = key.substr(split + 1),
                    .host_to_device = counts.host_to_device,
                    .device_to_host = counts.device_to_host,
                });
            }

            std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
                const size_t lhs_total = lhs.host_to_device + lhs.device_to_host;
                const size_t rhs_total = rhs.host_to_device + rhs.device_to_host;
                if (lhs_total != rhs_total)
                    return lhs_total > rhs_total;
                if (lhs.site != rhs.site)
                    return lhs.site < rhs.site;
                return lhs.parameter < rhs.parameter;
            });

            if (items.size() > limit)
                items.resize(limit);

            return items;
        }

      private:
        struct CopyCounts
        {
            size_t host_to_device = 0;
            size_t device_to_host = 0;
        };

        void record_copy_site(std::string_view site, bool host_to_device)
        {
            if (site.empty())
                return;

            std::lock_guard<std::mutex> lock(m_copy_site_mutex);
            auto& counts = m_copy_sites[std::string(site)];
            if (host_to_device)
                counts.host_to_device++;
            else
                counts.device_to_host++;
        }

        void record_copy_parameter(std::string_view site, std::string_view parameter, bool host_to_device)
        {
            if (site.empty() || parameter.empty())
                return;

            std::lock_guard<std::mutex> lock(m_copy_site_mutex);
            std::string key;
            key.reserve(site.size() + 1 + parameter.size());
            key.append(site);
            key.push_back('\n');
            key.append(parameter);

            auto& counts = m_copy_parameters[key];
            if (host_to_device)
                counts.host_to_device++;
            else
                counts.device_to_host++;
        }

        std::atomic<size_t> m_host_to_device_buffer_copies{0};
        std::atomic<size_t> m_device_to_host_buffer_copies{0};
        mutable std::mutex m_copy_site_mutex;
        std::unordered_map<std::string, CopyCounts> m_copy_sites;
        std::unordered_map<std::string, CopyCounts> m_copy_parameters;
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

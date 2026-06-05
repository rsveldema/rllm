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

#include <memory_spaces/host_memory_space.hpp>
#if defined(USE_VULKAN_OFFLOAD)
#include <memory_spaces/vulkan_memory_space.hpp>
#elif defined(USE_HIP_OFFLOAD)
#include <memory_spaces/hip_memory_space.hpp>
#endif
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

inline IMemorySpace* IMemorySpace::get_instance(IMemorySpaceType type)
{
    static HostMemorySpace host_instance;
#if defined(USE_VULKAN_OFFLOAD)
    static VulkanMemorySpace vulkan_instance;
#elif defined(USE_HIP_OFFLOAD)
    static HipMemorySpace hip_instance;
#endif

    switch (type)
    {
        case IMemorySpaceType::HOST:
            return &host_instance;
        case IMemorySpaceType::VULKAN:
#if defined(USE_VULKAN_OFFLOAD)
            return &vulkan_instance;
#else
            return get_instance();
#endif
        case IMemorySpaceType::HIP:
#if defined(USE_HIP_OFFLOAD)
            return &hip_instance;
#else
            return get_instance();
#endif
    }

    return get_instance();
}

template <typename T>
T* allocate_aligned(IMemorySpace& mem_space, std::size_t count)
{
    return reinterpret_cast<T*>(mem_space.allocate_staging(count * sizeof(T)));
}

template <typename T>
void deallocate_aligned(IMemorySpace& mem_space, T* ptr, std::size_t count)
{
    mem_space.release_staging(ptr, count * sizeof(T));
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
            size_t host_to_device_bytes = 0;
            size_t device_to_host_bytes = 0;
        };

        struct CopyParameterBreakdown
        {
            std::string site;
            std::string parameter;
            size_t host_to_device = 0;
            size_t device_to_host = 0;
            size_t host_to_device_bytes = 0;
            size_t device_to_host_bytes = 0;
        };

        void record_host_to_device_buffer_copy(std::string_view site = {}, std::string_view parameter = {}, size_t bytes = 0)
        {
#if defined(RLLM_ENABLE_STATISTICS)
            m_host_to_device_buffer_copies.fetch_add(1, std::memory_order_relaxed);
            m_host_to_device_buffer_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
            record_copy_site(site, true, bytes);
            record_copy_parameter(site, parameter, true, bytes);
#else
            static_cast<void>(site);
            static_cast<void>(parameter);
            static_cast<void>(bytes);
#endif
        }

        void record_device_to_host_buffer_copy(std::string_view site = {}, std::string_view parameter = {}, size_t bytes = 0)
        {
#if defined(RLLM_ENABLE_STATISTICS)
            m_device_to_host_buffer_copies.fetch_add(1, std::memory_order_relaxed);
            m_device_to_host_buffer_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
            record_copy_site(site, false, bytes);
            record_copy_parameter(site, parameter, false, bytes);
#else
            static_cast<void>(site);
            static_cast<void>(parameter);
            static_cast<void>(bytes);
#endif
        }

        size_t host_to_device_buffer_copies() const
        {
#if defined(RLLM_ENABLE_STATISTICS)
            return m_host_to_device_buffer_copies.load(std::memory_order_relaxed);
#else
            return 0;
#endif
        }

        size_t device_to_host_buffer_copies() const
        {
#if defined(RLLM_ENABLE_STATISTICS)
            return m_device_to_host_buffer_copies.load(std::memory_order_relaxed);
#else
            return 0;
#endif
        }

        size_t host_to_device_buffer_copy_bytes() const
        {
#if defined(RLLM_ENABLE_STATISTICS)
            return m_host_to_device_buffer_copy_bytes.load(std::memory_order_relaxed);
#else
            return 0;
#endif
        }

        size_t device_to_host_buffer_copy_bytes() const
        {
#if defined(RLLM_ENABLE_STATISTICS)
            return m_device_to_host_buffer_copy_bytes.load(std::memory_order_relaxed);
#else
            return 0;
#endif
        }

        void reset_buffer_copy_counters()
        {
#if defined(RLLM_ENABLE_STATISTICS)
            m_host_to_device_buffer_copies.store(0, std::memory_order_relaxed);
            m_device_to_host_buffer_copies.store(0, std::memory_order_relaxed);
            m_host_to_device_buffer_copy_bytes.store(0, std::memory_order_relaxed);
            m_device_to_host_buffer_copy_bytes.store(0, std::memory_order_relaxed);

            std::lock_guard<std::mutex> lock(m_copy_site_mutex);
            m_copy_sites.clear();
            m_copy_parameters.clear();
#endif
        }

        void print_statistics() const
        {
#if defined(RLLM_ENABLE_STATISTICS)
            std::println("Parallel statistics:");
            std::println(
                "  Host-to-device buffer copies: {} ({} bytes)",
                host_to_device_buffer_copies(),
                host_to_device_buffer_copy_bytes()
            );
            std::println(
                "  Device-to-host buffer copies: {} ({} bytes)",
                device_to_host_buffer_copies(),
                device_to_host_buffer_copy_bytes()
            );

            const auto top_sites = top_copy_sites();
            if (!top_sites.empty())
            {
                std::println("  Top copy sites:");
                for (const auto& site : top_sites)
                {
                    std::println(
                        "    {}: H2D={} ({} bytes), D2H={} ({} bytes)",
                        site.site,
                        site.host_to_device,
                        site.host_to_device_bytes,
                        site.device_to_host,
                        site.device_to_host_bytes
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
                        "    {} [{}]: H2D={} ({} bytes), D2H={} ({} bytes)",
                        item.site,
                        item.parameter,
                        item.host_to_device,
                        item.host_to_device_bytes,
                        item.device_to_host,
                        item.device_to_host_bytes
                    );
                }
            }
#else
            std::println("Parallel statistics: disabled");
#endif
        }

        std::vector<CopySiteBreakdown> top_copy_sites(size_t limit = 10) const
        {
#if defined(RLLM_ENABLE_STATISTICS)
            std::lock_guard<std::mutex> lock(m_copy_site_mutex);

            std::vector<CopySiteBreakdown> sites;
            sites.reserve(m_copy_sites.size());
            for (const auto& [site, counts] : m_copy_sites)
            {
                sites.push_back(CopySiteBreakdown{
                    .site = site,
                    .host_to_device = counts.host_to_device,
                    .device_to_host = counts.device_to_host,
                    .host_to_device_bytes = counts.host_to_device_bytes,
                    .device_to_host_bytes = counts.device_to_host_bytes,
                });
            }

            std::sort(sites.begin(), sites.end(), [](const auto& lhs, const auto& rhs) {
                const size_t lhs_total = lhs.host_to_device_bytes + lhs.device_to_host_bytes;
                const size_t rhs_total = rhs.host_to_device_bytes + rhs.device_to_host_bytes;
                if (lhs_total != rhs_total)
                    return lhs_total > rhs_total;
                const size_t lhs_count = lhs.host_to_device + lhs.device_to_host;
                const size_t rhs_count = rhs.host_to_device + rhs.device_to_host;
                if (lhs_count != rhs_count)
                    return lhs_count > rhs_count;
                return lhs.site < rhs.site;
            });

            if (sites.size() > limit)
                sites.resize(limit);

            return sites;
#else
            static_cast<void>(limit);
            return {};
#endif
        }

        std::vector<CopyParameterBreakdown> top_copy_parameters(size_t limit = 10) const
        {
#if defined(RLLM_ENABLE_STATISTICS)
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
                    .host_to_device_bytes = counts.host_to_device_bytes,
                    .device_to_host_bytes = counts.device_to_host_bytes,
                });
            }

            std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
                const size_t lhs_total = lhs.host_to_device_bytes + lhs.device_to_host_bytes;
                const size_t rhs_total = rhs.host_to_device_bytes + rhs.device_to_host_bytes;
                if (lhs_total != rhs_total)
                    return lhs_total > rhs_total;
                const size_t lhs_count = lhs.host_to_device + lhs.device_to_host;
                const size_t rhs_count = rhs.host_to_device + rhs.device_to_host;
                if (lhs_count != rhs_count)
                    return lhs_count > rhs_count;
                if (lhs.site != rhs.site)
                    return lhs.site < rhs.site;
                return lhs.parameter < rhs.parameter;
            });

            if (items.size() > limit)
                items.resize(limit);

            return items;
#else
            static_cast<void>(limit);
            return {};
#endif
        }

      private:
        struct CopyCounts
        {
            size_t host_to_device = 0;
            size_t device_to_host = 0;
            size_t host_to_device_bytes = 0;
            size_t device_to_host_bytes = 0;
        };

        void record_copy_site(std::string_view site, bool host_to_device, size_t bytes)
        {
#if defined(RLLM_ENABLE_STATISTICS)
            if (site.empty())
                return;

            std::lock_guard<std::mutex> lock(m_copy_site_mutex);
            auto& counts = m_copy_sites[std::string(site)];
            if (host_to_device)
            {
                counts.host_to_device++;
                counts.host_to_device_bytes += bytes;
            }
            else
            {
                counts.device_to_host++;
                counts.device_to_host_bytes += bytes;
            }
#else
            static_cast<void>(site);
            static_cast<void>(host_to_device);
            static_cast<void>(bytes);
#endif
        }

        void record_copy_parameter(std::string_view site, std::string_view parameter, bool host_to_device, size_t bytes)
        {
#if defined(RLLM_ENABLE_STATISTICS)
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
            {
                counts.host_to_device++;
                counts.host_to_device_bytes += bytes;
            }
            else
            {
                counts.device_to_host++;
                counts.device_to_host_bytes += bytes;
            }
#else
            static_cast<void>(site);
            static_cast<void>(parameter);
            static_cast<void>(host_to_device);
            static_cast<void>(bytes);
#endif
        }

        std::atomic<size_t> m_host_to_device_buffer_copies{0};
        std::atomic<size_t> m_device_to_host_buffer_copies{0};
        std::atomic<size_t> m_host_to_device_buffer_copy_bytes{0};
        std::atomic<size_t> m_device_to_host_buffer_copy_bytes{0};
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

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>

#include <vulkan_session.hpp>

namespace rllm::vulkan_runtime
{
    void set_session(VulkanSession& session);
    VulkanSession& session();
    VulkanComputeContext& context();
    std::recursive_mutex& mutex();

    /** Return the VulkanQueue at the given index.
     *  Queue 0 is the main queue (used by the main thread).
     *  Queues 1..N are used by parallel sections. */
    VulkanQueue& get_queue(size_t index);
    size_t queue_count();

    template <typename T>
    VBaseDeviceBuffer& device_buffer(T& value)
    {
        value.copy_to_offload_buffer();
        return value.device_buffer();
    }

    template <typename T>
    const VBaseDeviceBuffer& device_buffer(const T& value)
    {
        auto& mutable_value = const_cast<T&>(value);
        mutable_value.copy_to_offload_buffer();
        return mutable_value.device_buffer();
    }

    template <typename T>
    VBaseDeviceBuffer& device_buffer(T& value, size_t index)
    {
        auto& element = value[index];
        element.copy_to_offload_buffer();
        return element.device_buffer();
    }

    template <typename T>
    const VBaseDeviceBuffer& device_buffer(const T& value, size_t index)
    {
        auto& element = const_cast<T&>(value)[index];
        element.copy_to_offload_buffer();
        return element.device_buffer();
    }

    template <typename T>
    void mark_device_latest(T& value)
    {
        value.mark_device_latest();
    }

    template <typename T>
    void mark_device_latest(T& value, size_t index)
    {
        value[index].mark_device_latest();
    }
}

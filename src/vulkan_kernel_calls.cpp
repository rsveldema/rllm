#include <vulkan_kernel_calls.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <print>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include <parallel.hpp>

namespace rllm::vulkan
{
	namespace detail
	{
		static bool resolve_offset(const void* base, size_t total_size, const void* ptr, size_t bytes, VkDeviceSize& out_offset)
		{
			if (base == nullptr || ptr == nullptr || bytes > total_size)
				return false;

			const auto* base_bytes = static_cast<const std::byte*>(base);
			const auto* ptr_bytes = static_cast<const std::byte*>(ptr);
			if (ptr_bytes < base_bytes)
				return false;

			const auto offset = static_cast<size_t>(ptr_bytes - base_bytes);
			if (offset > total_size || bytes > total_size - offset)
				return false;

			out_offset = static_cast<VkDeviceSize>(offset);
			return true;
		}

		struct KernelLaunchCounterRegistry
		{
			std::mutex mutex;
			std::unordered_map<std::string, size_t> counts;
			bool exit_printer_registered = false;
		};

		static KernelLaunchCounterRegistry& kernel_launch_counter_registry()
		{
			static auto* registry = new KernelLaunchCounterRegistry();
			return *registry;
		}

		static void print_kernel_launch_counts()
		{
			auto& registry = kernel_launch_counter_registry();
			std::vector<std::pair<std::string, size_t>> counts;
			{
				std::lock_guard<std::mutex> lock(registry.mutex);
				counts.reserve(registry.counts.size());
				for (const auto& item : registry.counts)
					counts.push_back(item);
			}

			if (counts.empty())
				return;

			std::sort(counts.begin(), counts.end(), [](const auto& lhs, const auto& rhs) {
				if (lhs.second != rhs.second)
					return lhs.second > rhs.second;
				return lhs.first < rhs.first;
			});

			if (counts.size() > 10)
				counts.resize(10);

			std::println("Top ComputeKernel launch counts:");
			for (const auto& [kernel_name, count] : counts)
				std::println("  {}: {}", kernel_name, count);
		}

		static void record_kernel_launch(std::string_view kernel_name)
		{
			auto& registry = kernel_launch_counter_registry();
			std::lock_guard<std::mutex> lock(registry.mutex);
			if (!registry.exit_printer_registered)
			{
				std::atexit(print_kernel_launch_counts);
				registry.exit_printer_registered = true;
			}
			registry.counts[std::string(kernel_name)]++;
		}
	}

	ComputeKernelRuntime::ComputeKernelRuntime(std::string_view kernel_name, std::filesystem::path spirv_path)
			: m_name(kernel_name)
			, m_spirv_path(std::move(spirv_path))
	{
		if (m_spirv_path.is_relative())
		{
			m_spirv_path = std::filesystem::path(RLLM_VULKAN_KERNEL_ROOT) / m_spirv_path;
		}
		m_spirv_words = detail::load_spirv_words(m_spirv_path, m_name.c_str());
		m_parsed_ssbo_binding_count = detail::count_ssbo_bindings_in_glsl(m_spirv_path);
		m_local_size = detail::parse_local_size_from_glsl(m_spirv_path);
	}

	ComputeKernelRuntime::~ComputeKernelRuntime()
	{
		wait_for_in_flight_submit();

		if (m_submit_fence != VK_NULL_HANDLE && m_submit_fence_device != VK_NULL_HANDLE)
		{
			vkDestroyFence(m_submit_fence_device, m_submit_fence, nullptr);
			m_submit_fence = VK_NULL_HANDLE;
		}
		if (m_command_buffer != VK_NULL_HANDLE && m_command_buffer_device != VK_NULL_HANDLE &&
			m_command_pool != VK_NULL_HANDLE)
		{
			vkFreeCommandBuffers(m_command_buffer_device, m_command_pool, 1, &m_command_buffer);
			m_command_buffer = VK_NULL_HANDLE;
		}
		if (m_pipeline != VK_NULL_HANDLE && m_cached_device != VK_NULL_HANDLE)
		{
			vkDestroyPipeline(m_cached_device, m_pipeline, nullptr);
			m_pipeline = VK_NULL_HANDLE;
		}
		if (m_pipeline_layout != VK_NULL_HANDLE && m_cached_device != VK_NULL_HANDLE)
		{
			vkDestroyPipelineLayout(m_cached_device, m_pipeline_layout, nullptr);
			m_pipeline_layout = VK_NULL_HANDLE;
		}
		if (m_descriptor_set_layout != VK_NULL_HANDLE && m_cached_device != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorSetLayout(m_cached_device, m_descriptor_set_layout, nullptr);
			m_descriptor_set_layout = VK_NULL_HANDLE;
		}
		if (m_descriptor_pool != VK_NULL_HANDLE && m_cached_device != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorPool(m_cached_device, m_descriptor_pool, nullptr);
			m_descriptor_pool = VK_NULL_HANDLE;
			m_descriptor_set = VK_NULL_HANDLE;
		}
		if (m_shader_module != VK_NULL_HANDLE && m_cached_device != VK_NULL_HANDLE)
		{
			vkDestroyShaderModule(m_cached_device, m_shader_module, nullptr);
			m_shader_module = VK_NULL_HANDLE;
		}
	}

	void ComputeKernelRuntime::wait_for_in_flight_submit()
	{
		if (!m_submit_in_flight)
			return;

		if (m_submit_fence == VK_NULL_HANDLE || m_submit_fence_device == VK_NULL_HANDLE)
		{
			m_submit_in_flight = false;
			return;
		}

		const VkResult wait_result = vkWaitForFences(m_submit_fence_device, 1, &m_submit_fence, VK_TRUE, UINT64_MAX);
		if (wait_result != VK_SUCCESS)
		{
			LOG_ERROR(
				"vkWaitForFences failed for kernel '{}' ({}): {}",
				name(),
				spirv_path().string(),
				static_cast<int>(wait_result)
			);
			std::abort();
		}
		m_submit_in_flight = false;
	}

	void ComputeKernelRuntime::ensure_pipeline(VkDevice device, uint32_t ssbo_binding_count)
	{
		if (m_pipeline != VK_NULL_HANDLE && m_cached_device == device && m_ssbo_binding_count == ssbo_binding_count)
		{
			return;
		}

		if (m_cached_device != VK_NULL_HANDLE && (m_cached_device != device || m_ssbo_binding_count != ssbo_binding_count))
		{
			if (m_submit_fence != VK_NULL_HANDLE && m_submit_fence_device != VK_NULL_HANDLE)
			{
				vkDestroyFence(m_submit_fence_device, m_submit_fence, nullptr);
				m_submit_fence = VK_NULL_HANDLE;
				m_submit_fence_device = VK_NULL_HANDLE;
			}
			if (m_pipeline != VK_NULL_HANDLE)
			{
				vkDestroyPipeline(m_cached_device, m_pipeline, nullptr);
			}
			if (m_pipeline_layout != VK_NULL_HANDLE)
			{
				vkDestroyPipelineLayout(m_cached_device, m_pipeline_layout, nullptr);
			}
			if (m_descriptor_set_layout != VK_NULL_HANDLE)
			{
				vkDestroyDescriptorSetLayout(m_cached_device, m_descriptor_set_layout, nullptr);
			}
			if (m_descriptor_pool != VK_NULL_HANDLE)
			{
				vkDestroyDescriptorPool(m_cached_device, m_descriptor_pool, nullptr);
			}
			if (m_shader_module != VK_NULL_HANDLE)
			{
				vkDestroyShaderModule(m_cached_device, m_shader_module, nullptr);
			}
			m_shader_module = VK_NULL_HANDLE;
			m_pipeline_layout = VK_NULL_HANDLE;
			m_descriptor_set_layout = VK_NULL_HANDLE;
			m_descriptor_pool = VK_NULL_HANDLE;
			m_descriptor_set = VK_NULL_HANDLE;
			m_pipeline = VK_NULL_HANDLE;
			m_cached_device = VK_NULL_HANDLE;
			m_ssbo_binding_count = 0;
		}

		if (m_shader_module == VK_NULL_HANDLE)
		{
			VkShaderModuleCreateInfo shader_info{};
			shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			shader_info.codeSize = m_spirv_words.size() * sizeof(uint32_t);
			shader_info.pCode = m_spirv_words.data();

			const VkResult shader_result = vkCreateShaderModule(device, &shader_info, nullptr, &m_shader_module);
			if (shader_result != VK_SUCCESS)
			{
				LOG_ERROR(
					"vkCreateShaderModule failed for kernel '{}' ({}): {}",
					m_name,
					m_spirv_path.string(),
					static_cast<int>(shader_result)
				);
				std::abort();
			}
		}

		if (m_descriptor_set_layout == VK_NULL_HANDLE && ssbo_binding_count > 0)
		{
			std::vector<VkDescriptorSetLayoutBinding> bindings;
			bindings.reserve(ssbo_binding_count);
			for (uint32_t i = 0; i < ssbo_binding_count; ++i)
			{
				VkDescriptorSetLayoutBinding binding{};
				binding.binding = i;
				binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				binding.descriptorCount = 1;
				binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				bindings.push_back(binding);
			}

			VkDescriptorSetLayoutCreateInfo set_layout_info{};
			set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
			set_layout_info.pBindings = bindings.data();

			const VkResult set_layout_result =
				vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &m_descriptor_set_layout);
			if (set_layout_result != VK_SUCCESS)
			{
				LOG_ERROR(
					"vkCreateDescriptorSetLayout failed for kernel '{}' ({}): {}",
					m_name,
					m_spirv_path.string(),
					static_cast<int>(set_layout_result)
				);
				std::abort();
			}

			VkDescriptorPoolSize pool_size{};
			pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			pool_size.descriptorCount = ssbo_binding_count;

			VkDescriptorPoolCreateInfo pool_info{};
			pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			pool_info.maxSets = 1;
			pool_info.poolSizeCount = 1;
			pool_info.pPoolSizes = &pool_size;

			const VkResult pool_result = vkCreateDescriptorPool(device, &pool_info, nullptr, &m_descriptor_pool);
			if (pool_result != VK_SUCCESS)
			{
				LOG_ERROR(
					"vkCreateDescriptorPool failed for kernel '{}' ({}): {}",
					m_name,
					m_spirv_path.string(),
					static_cast<int>(pool_result)
				);
				std::abort();
			}

			VkDescriptorSetAllocateInfo alloc_info{};
			alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			alloc_info.descriptorPool = m_descriptor_pool;
			alloc_info.descriptorSetCount = 1;
			alloc_info.pSetLayouts = &m_descriptor_set_layout;

			const VkResult alloc_set_result = vkAllocateDescriptorSets(device, &alloc_info, &m_descriptor_set);
			if (alloc_set_result != VK_SUCCESS)
			{
				LOG_ERROR(
					"vkAllocateDescriptorSets failed for kernel '{}' ({}): {}",
					m_name,
					m_spirv_path.string(),
					static_cast<int>(alloc_set_result)
				);
				std::abort();
			}
		}

		if (m_pipeline_layout == VK_NULL_HANDLE)
		{
			VkPipelineLayoutCreateInfo layout_info{};
			layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			layout_info.pushConstantRangeCount = 1;
			VkPushConstantRange push_constant_range{};
			push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			push_constant_range.offset = 0;
			push_constant_range.size = 128;
			if (ssbo_binding_count > 0)
			{
				layout_info.setLayoutCount = 1;
				layout_info.pSetLayouts = &m_descriptor_set_layout;
			}
			layout_info.pPushConstantRanges = &push_constant_range;

			const VkResult layout_result = vkCreatePipelineLayout(device, &layout_info, nullptr, &m_pipeline_layout);
			if (layout_result != VK_SUCCESS)
			{
				vkDestroyShaderModule(device, m_shader_module, nullptr);
				m_shader_module = VK_NULL_HANDLE;
				LOG_ERROR(
					"vkCreatePipelineLayout failed for kernel '{}' ({}): {}",
					m_name,
					m_spirv_path.string(),
					static_cast<int>(layout_result)
				);
				std::abort();
			}
		}

		if (m_pipeline == VK_NULL_HANDLE)
		{
			VkPipelineShaderStageCreateInfo stage_info{};
			stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			stage_info.module = m_shader_module;
			stage_info.pName = "main";

			VkComputePipelineCreateInfo pipeline_info{};
			pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pipeline_info.stage = stage_info;
			pipeline_info.layout = m_pipeline_layout;

			const VkResult pipeline_result =
				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &m_pipeline);
			if (pipeline_result != VK_SUCCESS)
			{
				vkDestroyPipelineLayout(device, m_pipeline_layout, nullptr);
				vkDestroyShaderModule(device, m_shader_module, nullptr);
				m_pipeline_layout = VK_NULL_HANDLE;
				m_shader_module = VK_NULL_HANDLE;
				LOG_ERROR(
					"vkCreateComputePipelines failed for kernel '{}' ({}): {}",
					m_name,
					m_spirv_path.string(),
					static_cast<int>(pipeline_result)
				);
				std::abort();
			}
		}

		m_cached_device = device;
		m_ssbo_binding_count = ssbo_binding_count;
	}

	void ComputeKernelRuntime::dispatch_kernel(
		uint32_t groups_x,
		uint32_t groups_y,
		uint32_t groups_z,
		std::span<const detail::HostBufferView> buffers,
		std::span<const std::byte> push_constants,
		std::span<ComputeKernelRuntime::RuntimeBuffer> runtime_buffers,
		size_t& runtime_buffer_count
	)
	{
		if (groups_x == 0 || groups_y == 0 || groups_z == 0)
		{
			return;
		}

		detail::record_kernel_launch(name());

		detail::RuntimeContext& ctx = detail::runtime_context();
		std::lock_guard<std::mutex> runtime_lock(*ctx.launch_mutex);
		wait_for_in_flight_submit();
		const uint32_t ssbo_binding_count =
			std::max<uint32_t>(m_parsed_ssbo_binding_count, static_cast<uint32_t>(buffers.size()));
		if (buffers.size() != ssbo_binding_count)
		{
			LOG_ERROR(
				"Kernel '{}' expected {} SSBO buffers, but launch provided {}.",
				name(),
				ssbo_binding_count,
				buffers.size()
			);
			std::abort();
		}

		ensure_pipeline(ctx.device, ssbo_binding_count);

		if (runtime_buffers.size() < buffers.size())
		{
			LOG_ERROR(
				"Kernel '{}' needs {} runtime buffers, but storage only has {}.",
				name(),
				buffers.size(),
				runtime_buffers.size()
			);
			std::abort();
		}

		runtime_buffer_count = buffers.size();

		// Release a single runtime buffer slot (unmap + free).
		auto release_runtime_buffer = [&](RuntimeBuffer& rb) {
			if (rb.cached)
			{
				rb.view = {};
				rb.buffer = VK_NULL_HANDLE;
				rb.memory = VK_NULL_HANDLE;
				rb.staging_buffer = VK_NULL_HANDLE;
				rb.staging_memory = VK_NULL_HANDLE;
				rb.mapped = nullptr;
				rb.cached = false;
				rb.use_offload_source = false;
				rb.bind_offload_direct = false;
				rb.offload_source_offset = 0;
				return;
			}
			if (rb.mapped != nullptr)
			{
				vkUnmapMemory(ctx.device, rb.staging_memory);
				rb.mapped = nullptr;
			}
			if (rb.staging_buffer != VK_NULL_HANDLE)
			{
				vkDestroyBuffer(ctx.device, rb.staging_buffer, nullptr);
				rb.staging_buffer = VK_NULL_HANDLE;
			}
			if (rb.staging_memory != VK_NULL_HANDLE)
			{
				vkFreeMemory(ctx.device, rb.staging_memory, nullptr);
				rb.staging_memory = VK_NULL_HANDLE;
			}
			if (rb.buffer != VK_NULL_HANDLE)
			{
				vkDestroyBuffer(ctx.device, rb.buffer, nullptr);
				rb.buffer = VK_NULL_HANDLE;
			}
			if (rb.memory != VK_NULL_HANDLE)
			{
				vkFreeMemory(ctx.device, rb.memory, nullptr);
				rb.memory = VK_NULL_HANDLE;
			}
			rb.cached = false;
			rb.use_offload_source = false;
			rb.bind_offload_direct = false;
			rb.offload_source_offset = 0;
		};

		auto cleanup_runtime_buffers = [&]() {
			for (size_t i = 0; i < runtime_buffer_count; ++i)
			{
				RuntimeBuffer& rb = runtime_buffers[i];
				if (rb.cached)
					continue;  // cached runtime buffer — kept alive for deferred host sync
				if (rb.mapped != nullptr)
				{
					vkUnmapMemory(ctx.device, rb.staging_memory);
					rb.mapped = nullptr;
				}
				if (rb.staging_buffer != VK_NULL_HANDLE)
				{
					vkDestroyBuffer(ctx.device, rb.staging_buffer, nullptr);
					rb.staging_buffer = VK_NULL_HANDLE;
				}
				if (rb.staging_memory != VK_NULL_HANDLE)
				{
					vkFreeMemory(ctx.device, rb.staging_memory, nullptr);
					rb.staging_memory = VK_NULL_HANDLE;
				}
				if (rb.buffer != VK_NULL_HANDLE)
				{
					vkDestroyBuffer(ctx.device, rb.buffer, nullptr);
					rb.buffer = VK_NULL_HANDLE;
				}
				if (rb.memory != VK_NULL_HANDLE)
				{
					vkFreeMemory(ctx.device, rb.memory, nullptr);
					rb.memory = VK_NULL_HANDLE;
				}
			}
			runtime_buffer_count = 0;
		};

		auto find_memory_type_index = [&](uint32_t type_filter, VkMemoryPropertyFlags properties) -> uint32_t {
			VkPhysicalDeviceMemoryProperties mem_props{};
			vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &mem_props);
			for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
			{
				if ((type_filter & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
				{
					return i;
				}
			}
			LOG_ERROR("No compatible Vulkan memory type found for storage buffer allocation.");
			std::abort();
		};

		auto create_runtime_buffer_resource = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& out_buffer, VkDeviceMemory& out_memory) {
			VkBufferCreateInfo buffer_info{};
			buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer_info.size = size;
			buffer_info.usage = usage;
			buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			const VkResult create_buffer_result = vkCreateBuffer(ctx.device, &buffer_info, nullptr, &out_buffer);
			if (create_buffer_result != VK_SUCCESS)
			{
				LOG_ERROR("vkCreateBuffer failed for kernel '{}': {}", name(), static_cast<int>(create_buffer_result));
				cleanup_runtime_buffers();
				std::abort();
			}

			VkMemoryRequirements mem_req{};
			vkGetBufferMemoryRequirements(ctx.device, out_buffer, &mem_req);

			VkMemoryAllocateInfo alloc_info{};
			alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			alloc_info.allocationSize = mem_req.size;
			alloc_info.memoryTypeIndex = find_memory_type_index(mem_req.memoryTypeBits, properties);

			const VkResult alloc_result = vkAllocateMemory(ctx.device, &alloc_info, nullptr, &out_memory);
			if (alloc_result != VK_SUCCESS)
			{
				LOG_ERROR("vkAllocateMemory failed for kernel '{}': {}", name(), static_cast<int>(alloc_result));
				cleanup_runtime_buffers();
				std::abort();
			}

			const VkResult bind_result = vkBindBufferMemory(ctx.device, out_buffer, out_memory, 0);
			if (bind_result != VK_SUCCESS)
			{
				LOG_ERROR("vkBindBufferMemory failed for kernel '{}': {}", name(), static_cast<int>(bind_result));
				cleanup_runtime_buffers();
				std::abort();
			}
		};

		for (size_t buf_idx = 0; buf_idx < buffers.size(); ++buf_idx)
		{
			const detail::HostBufferView& view = buffers[buf_idx];
			RuntimeBuffer& rb = runtime_buffers[buf_idx];
			const bool keep_runtime_buffer = view.vulkan_runtime_buffer != nullptr;
			auto seed_runtime_buffer = [&](RuntimeBuffer& runtime_buffer) {
				runtime_buffer.use_offload_source = false;
				runtime_buffer.bind_offload_direct = false;
				runtime_buffer.offload_source_offset = 0;
				const DeviceMemoryOwner owner = view.memory_owner ? view.memory_owner() : DeviceMemoryOwner::ON_HOST;
				if (view.offload_ptr != nullptr
					&& (owner == DeviceMemoryOwner::ON_DEVICE || owner == DeviceMemoryOwner::REPLICATED))
				{
					if (!detail::resolve_offset(
							ctx.offload_base,
							ctx.offload_size,
							view.offload_ptr,
							view.size_bytes,
							runtime_buffer.offload_source_offset
						))
					{
						LOG_ERROR(
							"Unable to resolve persistent offload offset for kernel '{}' parameter '{}'.",
							name(),
							view.parameter_name.empty() ? std::string_view{"<unnamed>"} : view.parameter_name
						);
						cleanup_runtime_buffers();
						std::abort();
					}
					runtime_buffer.use_offload_source = true;
					const VkDeviceSize offset_alignment =
						ctx.storage_buffer_offset_alignment == 0 ? 1 : ctx.storage_buffer_offset_alignment;
					runtime_buffer.bind_offload_direct =
						runtime_buffer.offload_source_offset % offset_alignment == 0;
					return;
				}

				parallel::statistics.record_host_to_device_buffer_copy(name(), view.parameter_name, view.size_bytes);
				std::memcpy(runtime_buffer.mapped, runtime_buffer.view.host_ptr, runtime_buffer.view.size_bytes);
				if (view.offload_ptr != nullptr && view.mark_device_latest)
				{
					IMemorySpace::get_instance()->copy_staging_to_offload(
						const_cast<void*>(view.offload_ptr),
						runtime_buffer.view.host_ptr,
						runtime_buffer.view.size_bytes
					);
					view.mark_device_latest();
				}
			};

			// Drop any previous binding for this slot before rebinding it.
			if (rb.cached || rb.buffer != VK_NULL_HANDLE)
			{
				release_runtime_buffer(rb);
			}

			rb = RuntimeBuffer{};
			rb.view = view;

			if (view.size_bytes == 0 || view.size_bytes == std::numeric_limits<size_t>::max())
			{
				LOG_ERROR(
					"Refusing to allocate Vulkan buffer for kernel '{}' parameter '{}' with invalid size {}.",
					name(),
					view.parameter_name.empty() ? std::string_view{"<unnamed>"} : view.parameter_name,
					view.size_bytes
				);
				cleanup_runtime_buffers();
				std::abort();
			}

			if (keep_runtime_buffer)
			{
				VulkanRuntimeBuffer& stored = *view.vulkan_runtime_buffer;
				if (stored.device == ctx.device && stored.buffer != VK_NULL_HANDLE && stored.memory != VK_NULL_HANDLE &&
					stored.staging_buffer != VK_NULL_HANDLE && stored.staging_memory != VK_NULL_HANDLE &&
					stored.mapped != nullptr && stored.size_bytes >= view.size_bytes)
				{
					rb.buffer = stored.buffer;
					rb.memory = stored.memory;
					rb.staging_buffer = stored.staging_buffer;
					rb.staging_memory = stored.staging_memory;
					rb.mapped = stored.mapped;
					rb.view = view;
					rb.cached = true;
					seed_runtime_buffer(rb);
					continue;
				}

				stored.release();
			}

			create_runtime_buffer_resource(
				static_cast<VkDeviceSize>(view.size_bytes),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				rb.buffer,
				rb.memory
			);
			create_runtime_buffer_resource(
				static_cast<VkDeviceSize>(view.size_bytes),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				rb.staging_buffer,
				rb.staging_memory
			);

			VkMemoryRequirements staging_mem_req{};
			vkGetBufferMemoryRequirements(ctx.device, rb.staging_buffer, &staging_mem_req);
			const VkResult map_result = vkMapMemory(ctx.device, rb.staging_memory, 0, staging_mem_req.size, 0, &rb.mapped);
			if (map_result != VK_SUCCESS)
			{
				LOG_ERROR("vkMapMemory failed for kernel '{}': {}", name(), static_cast<int>(map_result));
				cleanup_runtime_buffers();
				std::abort();
			}

			seed_runtime_buffer(rb);
			if (keep_runtime_buffer)
			{
				VulkanRuntimeBuffer& stored = *view.vulkan_runtime_buffer;
				stored.device = ctx.device;
				stored.buffer = rb.buffer;
				stored.memory = rb.memory;
				stored.staging_buffer = rb.staging_buffer;
				stored.staging_memory = rb.staging_memory;
				stored.mapped = rb.mapped;
				stored.size_bytes = rb.view.size_bytes;
				rb.cached = true;
			}
		}

		if (m_submit_fence == VK_NULL_HANDLE || m_submit_fence_device != ctx.device)
		{
			if (m_submit_fence != VK_NULL_HANDLE && m_submit_fence_device != VK_NULL_HANDLE)
			{
				vkDestroyFence(m_submit_fence_device, m_submit_fence, nullptr);
				m_submit_fence = VK_NULL_HANDLE;
			}

			VkFenceCreateInfo fence_info{};
			fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

			const VkResult fence_result = vkCreateFence(ctx.device, &fence_info, nullptr, &m_submit_fence);
			if (fence_result != VK_SUCCESS)
			{
				LOG_ERROR(
					"vkCreateFence failed for kernel '{}' ({}): {}",
					name(),
					spirv_path().string(),
					static_cast<int>(fence_result)
				);
				std::abort();
			}
			m_submit_fence_device = ctx.device;
		}

		if (m_command_buffer == VK_NULL_HANDLE)
		{
			VkCommandBufferAllocateInfo alloc_info{};
			alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			alloc_info.commandPool = ctx.command_pool;
			alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			alloc_info.commandBufferCount = 1;

			const VkResult alloc_result = vkAllocateCommandBuffers(ctx.device, &alloc_info, &m_command_buffer);
			if (alloc_result != VK_SUCCESS)
			{
				LOG_ERROR(
					"vkAllocateCommandBuffers failed for kernel '{}' ({}): {}",
					name(),
					spirv_path().string(),
					static_cast<int>(alloc_result)
				);
				std::abort();
			}

			m_command_buffer_device = ctx.device;
			m_command_pool = ctx.command_pool;
		}
		else
		{
			if (m_command_buffer_device != ctx.device || m_command_pool != ctx.command_pool)
			{
				vkFreeCommandBuffers(m_command_buffer_device, m_command_pool, 1, &m_command_buffer);
				m_command_buffer = VK_NULL_HANDLE;

				VkCommandBufferAllocateInfo alloc_info{};
				alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				alloc_info.commandPool = ctx.command_pool;
				alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				alloc_info.commandBufferCount = 1;

				const VkResult alloc_result = vkAllocateCommandBuffers(ctx.device, &alloc_info, &m_command_buffer);
				if (alloc_result != VK_SUCCESS)
				{
					LOG_ERROR(
						"vkAllocateCommandBuffers failed for kernel '{}' ({}): {}",
						name(),
						spirv_path().string(),
						static_cast<int>(alloc_result)
					);
					std::abort();
				}

				m_command_buffer_device = ctx.device;
				m_command_pool = ctx.command_pool;
			}

			const VkResult reset_result = vkResetCommandBuffer(m_command_buffer, 0);
			if (reset_result != VK_SUCCESS)
			{
				LOG_ERROR(
					"vkResetCommandBuffer failed for kernel '{}' ({}): {}",
					name(),
					spirv_path().string(),
					static_cast<int>(reset_result)
				);
				std::abort();
			}
		}

		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		const VkResult begin_result = vkBeginCommandBuffer(m_command_buffer, &begin_info);
		if (begin_result != VK_SUCCESS)
		{
			LOG_ERROR(
				"vkBeginCommandBuffer failed for kernel '{}' ({}): {}",
				name(),
				spirv_path().string(),
				static_cast<int>(begin_result)
			);
			std::abort();
		}

		vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline());

		if (!push_constants.empty())
		{
			vkCmdPushConstants(
				m_command_buffer,
				m_pipeline_layout,
				VK_SHADER_STAGE_COMPUTE_BIT,
				0,
				static_cast<uint32_t>(push_constants.size()),
				push_constants.data()
			);
		}

		if (runtime_buffer_count > 0)
		{
			if (ctx.offload_buffer != VK_NULL_HANDLE && ctx.offload_size != 0)
			{
				VkBufferMemoryBarrier offload_visibility_barrier{};
				offload_visibility_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
				offload_visibility_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
				offload_visibility_barrier.dstAccessMask =
					VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
				offload_visibility_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				offload_visibility_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				offload_visibility_barrier.buffer = ctx.offload_buffer;
				offload_visibility_barrier.offset = 0;
				offload_visibility_barrier.size = static_cast<VkDeviceSize>(ctx.offload_size);
				vkCmdPipelineBarrier(
					m_command_buffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0,
					0,
					nullptr,
					1,
					&offload_visibility_barrier,
					0,
					nullptr
				);
			}

			std::vector<VkBufferCopy> upload_regions;
			std::vector<VkBufferMemoryBarrier> upload_barriers;
			upload_regions.reserve(runtime_buffer_count);
			upload_barriers.reserve(runtime_buffer_count);
			for (size_t i = 0; i < runtime_buffer_count; ++i)
			{
				if (runtime_buffers[i].bind_offload_direct)
					continue;

				VkBufferCopy upload_region{};
				upload_region.size = static_cast<VkDeviceSize>(runtime_buffers[i].view.size_bytes);
				const VkBuffer upload_src_buffer =
					runtime_buffers[i].use_offload_source ? ctx.offload_buffer : runtime_buffers[i].staging_buffer;
				if (runtime_buffers[i].use_offload_source)
					upload_region.srcOffset = runtime_buffers[i].offload_source_offset;
				vkCmdCopyBuffer(m_command_buffer, upload_src_buffer, runtime_buffers[i].buffer, 1, &upload_region);
				upload_regions.push_back(upload_region);

				VkBufferMemoryBarrier upload_barrier{};
				upload_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
				upload_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				upload_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
				upload_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				upload_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				upload_barrier.buffer = runtime_buffers[i].buffer;
				upload_barrier.offset = 0;
				upload_barrier.size = static_cast<VkDeviceSize>(runtime_buffers[i].view.size_bytes);
				upload_barriers.push_back(upload_barrier);
			}
			if (!upload_barriers.empty())
			{
				vkCmdPipelineBarrier(
					m_command_buffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0,
					0,
					nullptr,
					static_cast<uint32_t>(upload_barriers.size()),
					upload_barriers.data(),
					0,
					nullptr
				);
			}

			std::vector<VkDescriptorBufferInfo> buffer_infos(runtime_buffer_count);
			std::vector<VkWriteDescriptorSet> writes(runtime_buffer_count);

			for (size_t i = 0; i < runtime_buffer_count; ++i)
			{
				buffer_infos[i].buffer = runtime_buffers[i].bind_offload_direct ? ctx.offload_buffer : runtime_buffers[i].buffer;
				buffer_infos[i].offset = runtime_buffers[i].bind_offload_direct ? runtime_buffers[i].offload_source_offset : 0;
				buffer_infos[i].range = static_cast<VkDeviceSize>(runtime_buffers[i].view.size_bytes);

				writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[i].dstSet = m_descriptor_set;
				writes[i].dstBinding = static_cast<uint32_t>(i);
				writes[i].dstArrayElement = 0;
				writes[i].descriptorCount = 1;
				writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				writes[i].pBufferInfo = &buffer_infos[i];
			}

			vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
			vkCmdBindDescriptorSets(
				m_command_buffer,
				VK_PIPELINE_BIND_POINT_COMPUTE,
				m_pipeline_layout,
				0,
				1,
				&m_descriptor_set,
				0,
				nullptr
			);
		}
		vkCmdDispatch(m_command_buffer, groups_x, groups_y, groups_z);

		if (runtime_buffer_count > 0)
		{
			std::vector<VkBufferMemoryBarrier> download_source_barriers;
			std::vector<size_t> host_download_indices;
			std::vector<size_t> offload_download_indices;
			std::vector<VkDeviceSize> offload_download_offsets;
			download_source_barriers.reserve(runtime_buffer_count);
			host_download_indices.reserve(runtime_buffer_count);
			offload_download_indices.reserve(runtime_buffer_count);
			offload_download_offsets.reserve(runtime_buffer_count);
			for (size_t i = 0; i < runtime_buffer_count; ++i)
			{
				if (!runtime_buffers[i].view.on_device_ready && !runtime_buffers[i].view.writes_to_buffer)
					continue;

				if (runtime_buffers[i].view.offload_ptr != nullptr && !runtime_buffers[i].bind_offload_direct)
				{
					VkDeviceSize offload_dst_offset = 0;
					if (!detail::resolve_offset(
							ctx.offload_base,
							ctx.offload_size,
							runtime_buffers[i].view.offload_ptr,
							runtime_buffers[i].view.size_bytes,
							offload_dst_offset
						))
					{
						LOG_ERROR(
							"Unable to resolve persistent offload destination offset for kernel '{}' parameter '{}'.",
							name(),
							runtime_buffers[i].view.parameter_name.empty()
								? std::string_view{"<unnamed>"}
								: runtime_buffers[i].view.parameter_name
						);
						cleanup_runtime_buffers();
						std::abort();
					}
					offload_download_indices.push_back(i);
					offload_download_offsets.push_back(offload_dst_offset);
				}
				if (runtime_buffers[i].view.on_device_ready)
					host_download_indices.push_back(i);
				if (runtime_buffers[i].bind_offload_direct && !runtime_buffers[i].view.on_device_ready)
					continue;

				VkBufferMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
				barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.buffer = runtime_buffers[i].buffer;
				barrier.offset = 0;
				barrier.size = static_cast<VkDeviceSize>(runtime_buffers[i].view.size_bytes);
				download_source_barriers.push_back(barrier);
			}

			if (!download_source_barriers.empty())
			{
				vkCmdPipelineBarrier(
					m_command_buffer,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					0,
					0,
					nullptr,
					static_cast<uint32_t>(download_source_barriers.size()),
					download_source_barriers.data(),
					0,
					nullptr
				);

				for (size_t idx : host_download_indices)
				{
					VkBufferCopy region{};
					region.size = static_cast<VkDeviceSize>(runtime_buffers[idx].view.size_bytes);
					vkCmdCopyBuffer(m_command_buffer, runtime_buffers[idx].buffer, runtime_buffers[idx].staging_buffer, 1, &region);
				}
				for (size_t copy_idx = 0; copy_idx < offload_download_indices.size(); ++copy_idx)
				{
					const size_t idx = offload_download_indices[copy_idx];
					VkBufferCopy region{};
					region.dstOffset = offload_download_offsets[copy_idx];
					region.size = static_cast<VkDeviceSize>(runtime_buffers[idx].view.size_bytes);
					vkCmdCopyBuffer(m_command_buffer, runtime_buffers[idx].buffer, ctx.offload_buffer, 1, &region);
				}
			}
		}

		const VkResult end_result = vkEndCommandBuffer(m_command_buffer);
		if (end_result != VK_SUCCESS)
		{
			LOG_ERROR(
				"vkEndCommandBuffer failed for kernel '{}' ({}): {}",
				name(),
				spirv_path().string(),
				static_cast<int>(end_result)
			);
			std::abort();
		}

		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &m_command_buffer;

		const VkResult reset_fence_result = vkResetFences(ctx.device, 1, &m_submit_fence);
		if (reset_fence_result != VK_SUCCESS)
		{
			LOG_ERROR(
				"vkResetFences failed for kernel '{}' ({}): {}",
				name(),
				spirv_path().string(),
				static_cast<int>(reset_fence_result)
			);
			std::abort();
		}

		const VkResult submit_result = vkQueueSubmit(ctx.queue, 1, &submit_info, m_submit_fence);
		if (submit_result != VK_SUCCESS)
		{
			LOG_ERROR(
				"vkQueueSubmit failed for kernel '{}' ({}): {}",
				name(),
				spirv_path().string(),
				static_cast<int>(submit_result)
			);
			std::abort();
		}
		m_submit_in_flight = true;

		bool needs_immediate_wait = false;
		for (size_t i = 0; i < runtime_buffer_count; ++i)
		{
			RuntimeBuffer& rb = runtime_buffers[i];
			if (rb.view.on_device_ready)
				needs_immediate_wait = true;
			if (!rb.cached && !rb.bind_offload_direct)
				needs_immediate_wait = true;
		}
		if (needs_immediate_wait)
			wait_for_in_flight_submit();

		for (size_t i = 0; i < runtime_buffer_count; ++i)
		{
			RuntimeBuffer& rb = runtime_buffers[i];
			const bool kernel_writes_buffer = rb.view.on_device_ready || rb.view.writes_to_buffer;
			if (kernel_writes_buffer)
			{
				if (rb.view.mark_device_latest)
					rb.view.mark_device_latest();
				if (rb.view.on_device_ready)
					rb.view.on_device_ready(rb.mapped, rb.view.size_bytes, name(), rb.view.parameter_name);
				rb.view.on_device_ready = nullptr;
			}
		}

		cleanup_runtime_buffers();
	}

		namespace detail
		{
			struct [[maybe_unused]] VulkanCandidate
		{
			VkPhysicalDevice device = VK_NULL_HANDLE;
			uint32_t queue_family_index = 0;
			VkPhysicalDeviceProperties props{};
		};

			[[maybe_unused]] static std::string to_lower_copy(std::string s)
		{
			std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return s;
		}

			[[maybe_unused]] static uint32_t vendor_id_from_name(const std::string& name)
		{
			const std::string lowered = to_lower_copy(name);
			if (lowered == "nvidia") return 0x10DE;
			if (lowered == "amd") return 0x1002;
			if (lowered == "intel") return 0x8086;
			if (lowered == "arm") return 0x13B5;
			if (lowered == "qualcomm") return 0x5143;
			if (lowered == "apple") return 0x106B;
			if (lowered == "google") return 0x1AE0;
			if (lowered == "mesa") return 0x10005;
			return 0;
		}

			[[maybe_unused]] static bool contains_case_insensitive(const std::string& haystack, const std::string& needle)
		{
			if (needle.empty())
				return true;
			return to_lower_copy(haystack).find(to_lower_copy(needle)) != std::string::npos;
		}

			[[maybe_unused]] static int device_score(const VulkanCandidate& c)
		{
			int score = 0;
			if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
			if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 500;
			if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) score += 250;
			const std::string name = to_lower_copy(c.props.deviceName);
			if (name.find("llvmpipe") != std::string::npos || name.find("lavapipe") != std::string::npos || name.find("software") != std::string::npos)
				score -= 10000;
			return score;
		}

			[[maybe_unused]] static const VulkanCandidate* pick_candidate(const std::vector<VulkanCandidate>& candidates)
		{
			if (candidates.empty())
				return nullptr;

			const char* index_env = std::getenv("RLLM_VULKAN_DEVICE_INDEX");
			if (index_env != nullptr && *index_env != '\0')
			{
				const long idx = std::strtol(index_env, nullptr, 10);
				if (idx >= 0 && static_cast<size_t>(idx) < candidates.size())
					return &candidates[static_cast<size_t>(idx)];
			}

			const char* vendor_env = std::getenv("RLLM_VULKAN_VENDOR");
			const char* name_env = std::getenv("RLLM_VULKAN_DEVICE_SUBSTRING");
			const uint32_t vendor_id = vendor_env ? vendor_id_from_name(vendor_env) : 0u;
			const std::string name_substring = name_env ? std::string(name_env) : std::string();

			if (vendor_id != 0u || !name_substring.empty())
			{
				for (const VulkanCandidate& c : candidates)
				{
					if (vendor_id != 0u && c.props.vendorID != vendor_id)
						continue;
					if (!contains_case_insensitive(c.props.deviceName, name_substring))
						continue;
					return &c;
				}
			}

			return &*std::max_element(candidates.begin(), candidates.end(), [](const VulkanCandidate& a, const VulkanCandidate& b) {
				return device_score(a) < device_score(b);
			});
		}

		uint32_t count_ssbo_bindings_in_glsl(const std::filesystem::path& spirv_path)
		{
			std::filesystem::path comp_path = spirv_path;
			comp_path.replace_extension(".comp");

			std::ifstream input(comp_path);
			if (!input)
			{
				return 0;
			}

			const std::regex binding_re(R"(layout\s*\(\s*std430\s*,\s*set\s*=\s*0\s*,\s*binding\s*=\s*([0-9]+)\s*\))");
			uint32_t max_binding = 0;
			bool found = false;
			std::string line;
			while (std::getline(input, line))
			{
				std::smatch m;
				if (std::regex_search(line, m, binding_re))
				{
					found = true;
					uint32_t b = static_cast<uint32_t>(std::stoul(m[1].str()));
					if (b > max_binding)
					{
						max_binding = b;
					}
				}
			}

			return found ? (max_binding + 1u) : 0u;
		}

		detail::LocalSize parse_local_size_from_glsl(const std::filesystem::path& spirv_path)
		{
			std::filesystem::path comp_path = spirv_path;
			comp_path.replace_extension(".comp");

			std::ifstream input(comp_path);
			if (!input)
			{
				return {};
			}

			const std::regex local_size_re(
				R"(layout\s*\(\s*local_size_x\s*=\s*([0-9]+)\s*,\s*local_size_y\s*=\s*([0-9]+)\s*,\s*local_size_z\s*=\s*([0-9]+)\s*\)\s*in\s*;)"
			);
			std::string line;
			while (std::getline(input, line))
			{
				std::smatch match;
				if (std::regex_search(line, match, local_size_re))
				{
					return detail::LocalSize{
						.x = static_cast<uint32_t>(std::stoul(match[1].str())),
						.y = static_cast<uint32_t>(std::stoul(match[2].str())),
						.z = static_cast<uint32_t>(std::stoul(match[3].str())),
					};
				}
			}

			return {};
		}

		RuntimeContext create_runtime_context()
		{
			RuntimeContext ctx{};
			auto& memory_space = static_cast<VulkanMemorySpace&>(*IMemorySpace::get_instance());
			ctx.instance = memory_space.instance();
			ctx.physical_device = memory_space.physical_device();
			ctx.device = memory_space.device();
			ctx.queue_family_index = memory_space.queue_family_index();
			ctx.queue = memory_space.queue();
			ctx.command_pool = memory_space.command_pool();
			ctx.offload_buffer = memory_space.offload_buffer();
			ctx.offload_base = memory_space.offload_base();
			ctx.offload_size = memory_space.offload_size();
			VkPhysicalDeviceProperties props{};
			vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
			ctx.storage_buffer_offset_alignment = props.limits.minStorageBufferOffsetAlignment;
			return ctx;
		}

		RuntimeContext& runtime_context()
		{
			static RuntimeContext ctx = create_runtime_context();
			return ctx;
		}

		std::vector<uint32_t> load_spirv_words(const std::filesystem::path& spirv, const char* kernel_name)
		{
			std::ifstream input(spirv, std::ios::binary | std::ios::ate);
			if (!input)
			{
				LOG_ERROR(
					"Failed to open SPIR-V kernel '{}' for kernel '{}' reading.",
					spirv.string(),
					kernel_name
				);
				std::abort();
			}

			const std::streamoff bytes = input.tellg();
			if (bytes <= 0 || (bytes % static_cast<std::streamoff>(sizeof(uint32_t))) != 0)
			{
				LOG_ERROR(
					"SPIR-V kernel '{}' for kernel '{}' has invalid byte size {}.",
					spirv.string(),
					kernel_name,
					static_cast<long long>(bytes)
				);
				std::abort();
			}

			input.seekg(0, std::ios::beg);
			std::vector<uint32_t> words(static_cast<size_t>(bytes) / sizeof(uint32_t));
			if (!input.read(reinterpret_cast<char*>(words.data()), bytes))
			{
				LOG_ERROR(
					"Failed to read SPIR-V kernel '{}' bytes for kernel '{}'.",
					spirv.string(),
					kernel_name
				);
				std::abort();
			}
			return words;
		}

	} // namespace detail

} // namespace rllm::vulkan

#include <vulkan_kernel_calls.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <regex>

namespace rllm::vulkan
{

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
	}

	ComputeKernelRuntime::~ComputeKernelRuntime()
	{
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

	void ComputeKernelRuntime::ensure_pipeline(VkDevice device, uint32_t ssbo_binding_count)
	{
		if (m_pipeline != VK_NULL_HANDLE && m_cached_device == device && m_ssbo_binding_count == ssbo_binding_count)
		{
			return;
		}

		if (m_cached_device != VK_NULL_HANDLE && (m_cached_device != device || m_ssbo_binding_count != ssbo_binding_count))
		{
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
			if (ssbo_binding_count > 0)
			{
				layout_info.setLayoutCount = 1;
				layout_info.pSetLayouts = &m_descriptor_set_layout;
			}

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
		std::span<const detail::HostBufferView> buffers,
		std::span<ComputeKernelRuntime::RuntimeBuffer> runtime_buffers,
		size_t& runtime_buffer_count
	)
	{
		if (groups_x == 0 || groups_y == 0)
		{
			return;
		}

		detail::RuntimeContext& ctx = detail::runtime_context();
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

		runtime_buffer_count = 0;

		auto cleanup_runtime_buffers = [&]() {
			for (size_t i = 0; i < runtime_buffer_count; ++i)
			{
				RuntimeBuffer& rb = runtime_buffers[i];
				if (rb.mapped != nullptr)
				{
					if (rb.view.writable)
					{
						std::memcpy(rb.view.host_ptr, rb.mapped, rb.view.size_bytes);
					}
					vkUnmapMemory(ctx.device, rb.memory);
					rb.mapped = nullptr;
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

		for (const detail::HostBufferView& view : buffers)
		{
			RuntimeBuffer rb{};
			rb.view = view;

			VkBufferCreateInfo buffer_info{};
			buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer_info.size = static_cast<VkDeviceSize>(view.size_bytes);
			buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			const VkResult create_buffer_result = vkCreateBuffer(ctx.device, &buffer_info, nullptr, &rb.buffer);
			if (create_buffer_result != VK_SUCCESS)
			{
				LOG_ERROR("vkCreateBuffer failed for kernel '{}': {}", name(), static_cast<int>(create_buffer_result));
				cleanup_runtime_buffers();
				std::abort();
			}

			VkMemoryRequirements mem_req{};
			vkGetBufferMemoryRequirements(ctx.device, rb.buffer, &mem_req);

			VkMemoryAllocateInfo alloc_info{};
			alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			alloc_info.allocationSize = mem_req.size;
			alloc_info.memoryTypeIndex =
				find_memory_type_index(mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			const VkResult alloc_result = vkAllocateMemory(ctx.device, &alloc_info, nullptr, &rb.memory);
			if (alloc_result != VK_SUCCESS)
			{
				LOG_ERROR("vkAllocateMemory failed for kernel '{}': {}", name(), static_cast<int>(alloc_result));
				cleanup_runtime_buffers();
				std::abort();
			}

			const VkResult bind_result = vkBindBufferMemory(ctx.device, rb.buffer, rb.memory, 0);
			if (bind_result != VK_SUCCESS)
			{
				LOG_ERROR("vkBindBufferMemory failed for kernel '{}': {}", name(), static_cast<int>(bind_result));
				cleanup_runtime_buffers();
				std::abort();
			}

			const VkResult map_result = vkMapMemory(ctx.device, rb.memory, 0, alloc_info.allocationSize, 0, &rb.mapped);
			if (map_result != VK_SUCCESS)
			{
				LOG_ERROR("vkMapMemory failed for kernel '{}': {}", name(), static_cast<int>(map_result));
				cleanup_runtime_buffers();
				std::abort();
			}

			std::memcpy(rb.mapped, rb.view.host_ptr, rb.view.size_bytes);
			runtime_buffers[runtime_buffer_count++] = rb;
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

		if (runtime_buffer_count > 0)
		{
			std::vector<VkDescriptorBufferInfo> buffer_infos(runtime_buffer_count);
			std::vector<VkWriteDescriptorSet> writes(runtime_buffer_count);

			for (size_t i = 0; i < runtime_buffer_count; ++i)
			{
				buffer_infos[i].buffer = runtime_buffers[i].buffer;
				buffer_infos[i].offset = 0;
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
		vkCmdDispatch(m_command_buffer, groups_x, groups_y, 1);

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

		const VkResult submit_result = vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE);
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

		const VkResult wait_result = vkQueueWaitIdle(ctx.queue);
		if (wait_result != VK_SUCCESS)
		{
			LOG_ERROR(
				"vkQueueWaitIdle failed for kernel '{}' ({}): {}",
				name(),
				spirv_path().string(),
				static_cast<int>(wait_result)
			);
			std::abort();
		}

		cleanup_runtime_buffers();
	}

	namespace detail
	{
		struct VulkanCandidate
		{
			VkPhysicalDevice device = VK_NULL_HANDLE;
			uint32_t queue_family_index = 0;
			VkPhysicalDeviceProperties props{};
		};

		static std::string to_lower_copy(std::string s)
		{
			std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return s;
		}

		static uint32_t vendor_id_from_name(const std::string& name)
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

		static bool contains_case_insensitive(const std::string& haystack, const std::string& needle)
		{
			if (needle.empty())
				return true;
			return to_lower_copy(haystack).find(to_lower_copy(needle)) != std::string::npos;
		}

		static int device_score(const VulkanCandidate& c)
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

		static const VulkanCandidate* pick_candidate(const std::vector<VulkanCandidate>& candidates)
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

		RuntimeContext create_runtime_context()
		{
			RuntimeContext ctx{};

			VkApplicationInfo app_info{};
			app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			app_info.pApplicationName = "rllm";
			app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
			app_info.pEngineName = "rllm";
			app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
			app_info.apiVersion = VK_API_VERSION_1_1;

			VkInstanceCreateInfo instance_info{};
			instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			instance_info.pApplicationInfo = &app_info;

			const VkResult instance_result = vkCreateInstance(&instance_info, nullptr, &ctx.instance);
			if (instance_result != VK_SUCCESS)
			{
				LOG_ERROR("vkCreateInstance failed: {}", static_cast<int>(instance_result));
				std::abort();
			}

			uint32_t physical_count = 0;
			const VkResult enum_result = vkEnumeratePhysicalDevices(ctx.instance, &physical_count, nullptr);
			if (enum_result != VK_SUCCESS || physical_count == 0)
			{
				LOG_ERROR("vkEnumeratePhysicalDevices failed or found no devices: {}", static_cast<int>(enum_result));
				std::abort();
			}

			std::vector<VkPhysicalDevice> physical_devices(physical_count);
			const VkResult enum_fill_result =
				vkEnumeratePhysicalDevices(ctx.instance, &physical_count, physical_devices.data());
			if (enum_fill_result != VK_SUCCESS)
			{
				LOG_ERROR("vkEnumeratePhysicalDevices (fill) failed: {}", static_cast<int>(enum_fill_result));
				std::abort();
			}

			std::vector<VulkanCandidate> candidates;
			for (const VkPhysicalDevice physical_device : physical_devices)
			{
				uint32_t queue_family_count = 0;
				vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
				if (queue_family_count == 0)
				{
					continue;
				}

				std::vector<VkQueueFamilyProperties> queue_props(queue_family_count);
				vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_props.data());

				for (uint32_t i = 0; i < queue_family_count; ++i)
				{
					if ((queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u)
					{
						VulkanCandidate c{};
						c.device = physical_device;
						c.queue_family_index = i;
						vkGetPhysicalDeviceProperties(physical_device, &c.props);
						candidates.push_back(c);
						break;
					}
				}
			}

			const VulkanCandidate* chosen = pick_candidate(candidates);
			if (chosen == nullptr)
			{
				LOG_ERROR("No Vulkan physical device with a compute-capable queue family was found.");
				std::abort();
			}

			ctx.physical_device = chosen->device;
			ctx.queue_family_index = chosen->queue_family_index;

			const float queue_priority = 1.0f;
			VkDeviceQueueCreateInfo queue_info{};
			queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queue_info.queueFamilyIndex = ctx.queue_family_index;
			queue_info.queueCount = 1;
			queue_info.pQueuePriorities = &queue_priority;

			VkDeviceCreateInfo device_info{};
			device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
			device_info.queueCreateInfoCount = 1;
			device_info.pQueueCreateInfos = &queue_info;

			const VkResult device_result = vkCreateDevice(ctx.physical_device, &device_info, nullptr, &ctx.device);
			if (device_result != VK_SUCCESS)
			{
				LOG_ERROR("vkCreateDevice failed: {}", static_cast<int>(device_result));
				std::abort();
			}

			vkGetDeviceQueue(ctx.device, ctx.queue_family_index, 0, &ctx.queue);
			if (ctx.queue == VK_NULL_HANDLE)
			{
				LOG_ERROR("vkGetDeviceQueue returned a null queue handle.");
				std::abort();
			}

			VkCommandPoolCreateInfo pool_info{};
			pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			pool_info.queueFamilyIndex = ctx.queue_family_index;

			const VkResult pool_result = vkCreateCommandPool(ctx.device, &pool_info, nullptr, &ctx.command_pool);
			if (pool_result != VK_SUCCESS)
			{
				LOG_ERROR("vkCreateCommandPool failed: {}", static_cast<int>(pool_result));
				std::abort();
			}

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

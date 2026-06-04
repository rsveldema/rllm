# ── Parallel backend selection ────────────────────────────────────────────────
# cmake -DPARALLEL_BACKEND=openmp|fastfork|sequential  (default: fastfork)
set(PARALLEL_BACKEND "fastfork" CACHE STRING "Parallel backend: openmp, fastfork, or sequential")
set_property(CACHE PARALLEL_BACKEND PROPERTY STRINGS openmp fastfork sequential)
set(OFFLOAD_BACKEND "none" CACHE STRING "Offload backend: none, hip, or vulkan")
set_property(CACHE OFFLOAD_BACKEND PROPERTY STRINGS none hip vulkan)
set(HIP_ROOT "/opt/rocm" CACHE PATH "HIP installation root")

include(CheckCXXSourceCompiles)

if(NOT PARALLEL_BACKEND MATCHES "^(openmp|fastfork|sequential)$")
    message(FATAL_ERROR "PARALLEL_BACKEND must be one of: openmp, fastfork, sequential (got '${PARALLEL_BACKEND}')")
endif()

if(NOT OFFLOAD_BACKEND MATCHES "^(none|hip|vulkan)$")
    message(FATAL_ERROR "OFFLOAD_BACKEND must be one of: none, hip, vulkan (got '${OFFLOAD_BACKEND}')")
endif()

if(PARALLEL_BACKEND STREQUAL "fastfork")
    message(STATUS "Parallel backend: fastfork")
elseif(PARALLEL_BACKEND STREQUAL "openmp")
    find_package(OpenMP REQUIRED)
    message(STATUS "Parallel backend: OpenMP")
else()
    message(STATUS "Parallel backend: sequential (single-threaded)")
endif()

if(OFFLOAD_BACKEND STREQUAL "hip")
    if(NOT EXISTS "${HIP_ROOT}/include")
        message(FATAL_ERROR "OFFLOAD_BACKEND=hip requires HIP_ROOT/include (set HIP_ROOT, current: '${HIP_ROOT}')")
    endif()
    set(CMAKE_REQUIRED_INCLUDES "${HIP_ROOT}/include")
    set(CMAKE_REQUIRED_DEFINITIONS "-D__HIP_PLATFORM_AMD__=1")
    check_cxx_source_compiles("#include <hip/hip_runtime.h>\nint main() { return 0; }" RLLM_HIP_HEADERS_OK)
    unset(CMAKE_REQUIRED_DEFINITIONS)
    unset(CMAKE_REQUIRED_INCLUDES)
    if(NOT RLLM_HIP_HEADERS_OK)
        message(FATAL_ERROR "OFFLOAD_BACKEND=hip requires a usable HIP header setup at HIP_ROOT='${HIP_ROOT}' (failed to compile a probe including <hip/hip_runtime.h> with __HIP_PLATFORM_AMD__).")
    endif()
    find_library(
        RLLM_HIP_RUNTIME_LIBRARY
        NAMES amdhip64
        HINTS "${HIP_ROOT}/lib" "${HIP_ROOT}/lib64" /usr/lib64 /lib64
    )
    if(NOT RLLM_HIP_RUNTIME_LIBRARY)
        message(FATAL_ERROR "OFFLOAD_BACKEND=hip requires libamdhip64 (HIP runtime), but it was not found. Set HIP_ROOT or install hip-runtime-amd.")
    endif()
    message(STATUS "HIP offload macros enabled (USE_HIP_OFFLOAD)")
    message(STATUS "Using HIP root: ${HIP_ROOT}")
    message(STATUS "HIP runtime library: ${RLLM_HIP_RUNTIME_LIBRARY}")
elseif(OFFLOAD_BACKEND STREQUAL "vulkan")
    find_package(Vulkan REQUIRED)
    find_package(VulkanMemoryAllocator CONFIG QUIET)
    if(NOT VulkanMemoryAllocator_FOUND)
        FetchContent_Declare(
            VulkanMemoryAllocator
            GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
            GIT_TAG        v3.3.0
        )
        FetchContent_MakeAvailable(VulkanMemoryAllocator)
    endif()
    if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
        message(FATAL_ERROR "Failed to provide GPUOpen::VulkanMemoryAllocator from either find_package or FetchContent.")
    endif()
    set(CMAKE_REQUIRED_INCLUDES "${Vulkan_INCLUDE_DIRS}")
    check_cxx_source_compiles(
        "#include <vulkan/vulkan.h>\nint main() { return VK_HEADER_VERSION; }"
        RLLM_VULKAN_HEADERS_OK
    )
    unset(CMAKE_REQUIRED_INCLUDES)
    if(NOT RLLM_VULKAN_HEADERS_OK)
        message(FATAL_ERROR "OFFLOAD_BACKEND=vulkan requires usable Vulkan headers (failed to compile a probe including <vulkan/vulkan.h>).")
    endif()
    find_program(RLLM_VULKAN_SHADER_COMPILER NAMES glslc glslangValidator)
    if(NOT RLLM_VULKAN_SHADER_COMPILER)
        message(FATAL_ERROR "OFFLOAD_BACKEND=vulkan requires glslc or glslangValidator to compile generated kernels to SPIR-V.")
    endif()
    message(STATUS "Vulkan offload mode enabled (USE_VULKAN_OFFLOAD)")
    message(STATUS "Vulkan includes: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan libraries: ${Vulkan_LIBRARIES}")
    message(STATUS "Vulkan shader compiler: ${RLLM_VULKAN_SHADER_COMPILER}")
endif()

set(RLLM_COMPILE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(RLLM_OFFLOAD_PREPROCESS_TARGET "")
file(
    GLOB_RECURSE RLLM_OFFLOAD_INPUT_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(OFFLOAD_BACKEND STREQUAL "hip")
    set(RLLM_HIPIFIED_SRC_DIR "${CMAKE_BINARY_DIR}/generated/hipified/src")
    set(RLLM_HIPIFY_MANIFEST "${CMAKE_BINARY_DIR}/generated/hipified/hipified_sources.cmake")
    set(RLLM_OFFLOAD_PREPROCESS_TARGET rllm_generate_hipified_sources)

    add_custom_target(${RLLM_OFFLOAD_PREPROCESS_TARGET}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${RLLM_HIPIFIED_SRC_DIR}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/hipify.py"
                --src-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --out-dir "${RLLM_HIPIFIED_SRC_DIR}"
                --enum-value-tool "$<TARGET_FILE:generate_domain_replace_json>"
                --manifest "${RLLM_HIPIFY_MANIFEST}"
        DEPENDS
            generate_domain_replace_json
            ${RLLM_OFFLOAD_INPUT_SOURCES}
            "${CMAKE_SOURCE_DIR}/hipify.py"
            "${CMAKE_SOURCE_DIR}/offloadize_common.py"
        COMMENT "Generating HIP-offload source tree from src/"
        VERBATIM
    )

    set(RLLM_COMPILE_SRC_DIR "${RLLM_HIPIFIED_SRC_DIR}")
endif()

if(OFFLOAD_BACKEND STREQUAL "vulkan")
    set(RLLM_VULKANIZED_SRC_DIR "${CMAKE_BINARY_DIR}/generated/vulkanized/src")
    set(RLLM_VULKAN_KERNEL_DIR "${CMAKE_BINARY_DIR}/generated/vulkanized/kernels")
    set(RLLM_VULKANIZE_MANIFEST "${CMAKE_BINARY_DIR}/generated/vulkanized/vulkanized_sources.cmake")
    set(RLLM_OFFLOAD_PREPROCESS_TARGET rllm_generate_vulkanized_sources)

    add_custom_target(${RLLM_OFFLOAD_PREPROCESS_TARGET}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${RLLM_VULKANIZED_SRC_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${RLLM_VULKAN_KERNEL_DIR}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/vulkanize.py"
                --src-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --out-dir "${RLLM_VULKANIZED_SRC_DIR}"
                --shader-compiler "${RLLM_VULKAN_SHADER_COMPILER}"
                --enum-value-tool "$<TARGET_FILE:generate_domain_replace_json>"
                --manifest "${RLLM_VULKANIZE_MANIFEST}"
        DEPENDS
            generate_domain_replace_json
            ${RLLM_OFFLOAD_INPUT_SOURCES}
            "${CMAKE_SOURCE_DIR}/vulkanize.py"
            "${CMAKE_SOURCE_DIR}/vulkan-config.json"
            "${CMAKE_SOURCE_DIR}/offloadize_common.py"
        COMMENT "Generating Vulkan-offload source tree from src/ and compiling shaders"
        VERBATIM
    )

    set(RLLM_COMPILE_SRC_DIR "${RLLM_VULKANIZED_SRC_DIR}")
endif()

# ── Helper to create an rllm static library variant ──────────────────────────
# Usage: rllm_define_lib(<target> <extra compile flags…>)
function(rllm_define_lib target_name)
    add_library(${target_name} STATIC)

    if(RLLM_OFFLOAD_PREPROCESS_TARGET)
        add_dependencies(${target_name} ${RLLM_OFFLOAD_PREPROCESS_TARGET})
    endif()

    set(RLLM_LIBRARY_SOURCES
        "${RLLM_COMPILE_SRC_DIR}/Trainer.cc"
        "${RLLM_COMPILE_SRC_DIR}/Prompter.cc"
        "${RLLM_COMPILE_SRC_DIR}/Corpus.cpp"
        "${RLLM_COMPILE_SRC_DIR}/InputLayer.cc"
        "${RLLM_COMPILE_SRC_DIR}/vecmath.cc"
        "${RLLM_COMPILE_SRC_DIR}/TransformerBlock.cc"
        "${RLLM_COMPILE_SRC_DIR}/OutputLayer.cc"
        "${RLLM_COMPILE_SRC_DIR}/NeuralNetwork.cc"
        "${RLLM_COMPILE_SRC_DIR}/serialization.cc"
        "${RLLM_COMPILE_SRC_DIR}/parallel.cc"
    )

    if(RLLM_OFFLOAD_PREPROCESS_TARGET)
        set_source_files_properties(${RLLM_LIBRARY_SOURCES} PROPERTIES GENERATED TRUE)
    endif()

    target_sources(${target_name} PRIVATE
        ${RLLM_LIBRARY_SOURCES}
        "${CMAKE_BINARY_DIR}/generated/tokenizer_map.cc"
    )

    target_include_directories(${target_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
    )

    # Warning flags common to all variants; optimization flags come from ARGN.
    target_compile_options(${target_name} PRIVATE
        -Wall -Wextra -Wpedantic -Wstack-usage=10000 -fopenmp-simd -Wno-narrowing -Wno-unused-parameter
        ${ARGN}
    )
    #target_compile_definitions(${target_name} PUBLIC RLLM_ENABLE_OVERFLOW_CHECK_ADD)

    add_dependencies(${target_name} generate_tokenizer_map)
    target_link_libraries(${target_name} PUBLIC nlohmann_json::nlohmann_json)

    if(PARALLEL_BACKEND STREQUAL "fastfork")
        set(RLLM_FASTFORK_SOURCE "${RLLM_COMPILE_SRC_DIR}/fastfork/fastfork.cc")
        if(RLLM_OFFLOAD_PREPROCESS_TARGET)
            set_source_files_properties("${RLLM_FASTFORK_SOURCE}" PROPERTIES GENERATED TRUE)
        endif()
        target_sources(${target_name} PRIVATE "${RLLM_FASTFORK_SOURCE}")
        target_compile_definitions(${target_name} PUBLIC USE_FASTFORK)
        target_link_libraries(${target_name} PUBLIC hwloc)
    elseif(PARALLEL_BACKEND STREQUAL "openmp")
        target_compile_definitions(${target_name} PUBLIC USE_OPENMP)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    if(OFFLOAD_BACKEND STREQUAL "hip")
        target_compile_definitions(${target_name} PUBLIC USE_HIP_OFFLOAD __HIP_PLATFORM_AMD__=1)
        target_include_directories(${target_name} PUBLIC ${HIP_ROOT}/include)
        target_link_libraries(${target_name} PUBLIC ${RLLM_HIP_RUNTIME_LIBRARY})
    elseif(OFFLOAD_BACKEND STREQUAL "vulkan")
        target_sources(${target_name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_kernel_calls.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_memory_space.cpp"
        )
        target_compile_definitions(
            ${target_name} PUBLIC
            USE_VULKAN_OFFLOAD
            RLLM_VULKAN_KERNEL_ROOT="${RLLM_VULKAN_KERNEL_DIR}"
        )
        target_link_libraries(${target_name} PUBLIC Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)
    endif()
endfunction()

# ── Library variants ──────────────────────────────────────────────────────────
rllm_define_lib(rllm_lib_opt   -march=native -O3)
rllm_define_lib(rllm_lib_debug -O0 -g)

# ── Dispatch: rllm_lib → opt for Release, debug for everything else ───────────
add_library(rllm_lib INTERFACE)
target_link_libraries(rllm_lib INTERFACE
    $<IF:$<CONFIG:Release>,rllm_lib_opt,rllm_lib_debug>
)

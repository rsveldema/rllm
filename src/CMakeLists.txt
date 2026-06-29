# ── Parallel backend selection ────────────────────────────────────────────────
# cmake -DPARALLEL_BACKEND=openmp|fastfork|sequential  (default: fastfork)
set(PARALLEL_BACKEND "fastfork" CACHE STRING "Parallel backend: openmp, fastfork, or sequential")
set_property(CACHE PARALLEL_BACKEND PROPERTY STRINGS openmp fastfork sequential)
set(OFFLOAD_BACKEND "none" CACHE STRING "Offload backend: none or vulkan")
set_property(CACHE OFFLOAD_BACKEND PROPERTY STRINGS none vulkan)
option(RLLM_ENABLE_STATISTICS "Enable runtime statistics gathering and reporting" ON)

include(CheckCXXSourceCompiles)

if(NOT PARALLEL_BACKEND MATCHES "^(openmp|fastfork|sequential)$")
    message(FATAL_ERROR "PARALLEL_BACKEND must be one of: openmp, fastfork, sequential (got '${PARALLEL_BACKEND}')")
endif()

if(NOT OFFLOAD_BACKEND MATCHES "^(none|vulkan)$")
    message(FATAL_ERROR "OFFLOAD_BACKEND must be one of: none, vulkan (got '${OFFLOAD_BACKEND}')")
endif()

if(PARALLEL_BACKEND STREQUAL "fastfork")
    message(STATUS "Parallel backend: fastfork")
elseif(PARALLEL_BACKEND STREQUAL "openmp")
    find_package(OpenMP REQUIRED)
    message(STATUS "Parallel backend: OpenMP")
else()
    message(STATUS "Parallel backend: sequential (single-threaded)")
endif()
message(STATUS "Runtime statistics: ${RLLM_ENABLE_STATISTICS}")

if(OFFLOAD_BACKEND STREQUAL "vulkan")
    find_package(Vulkan REQUIRED)
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
            "${CMAKE_SOURCE_DIR}/kernel_compiler/codegen/compile.py"
            "${CMAKE_SOURCE_DIR}/kernel_compiler/codegen/parser.py"
            "${CMAKE_SOURCE_DIR}/kernel_compiler/codegen/transforms.py"
            "${CMAKE_SOURCE_DIR}/kernel_compiler/codegen/visitors/tree_rewriter.py"
            "${CMAKE_SOURCE_DIR}/kernel_compiler/codegen/visitors/vulkan_kernel_visitor.py"
            "${CMAKE_SOURCE_DIR}/kernel_compiler/codegen/visitors/rllm_vulkan_dispatch_stub_visitor.py"
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
        "${RLLM_COMPILE_SRC_DIR}/RuntimeConfig.cc"
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
        ${CMAKE_SOURCE_DIR}/kernel_compiler/runtime/include
    )

    # Warning flags common to all variants; optimization flags come from ARGN.
    target_compile_options(${target_name} PRIVATE
        -Wall -Wextra -Wpedantic -Wstack-usage=10000 -fopenmp-simd -Wno-narrowing -Wno-unused-parameter
        ${ARGN}
    )
    #target_compile_definitions(${target_name} PUBLIC RLLM_ENABLE_OVERFLOW_CHECK_ADD)
    target_compile_definitions(${target_name} PUBLIC PREFER_SPEED_OVER_MEMORY=$<BOOL:${PREFER_SPEED_OVER_MEMORY}>)
    if(RLLM_ENABLE_STATISTICS)
        target_compile_definitions(${target_name} PUBLIC RLLM_ENABLE_STATISTICS)
    endif()

    add_dependencies(${target_name} generate_tokenizer_map)
    target_link_libraries(${target_name} PUBLIC nlohmann_json::nlohmann_json)
    target_link_libraries(${target_name} PUBLIC safetensors-cpp::safetensors_cpp)
    target_link_libraries(${target_name} PUBLIC isocline)

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

    if(OFFLOAD_BACKEND STREQUAL "vulkan")
        target_sources(${target_name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/rllm_vulkan_runtime.cpp"
            "${CMAKE_SOURCE_DIR}/kernel_compiler/runtime/src/vulkan_session.cc"
        )
        target_compile_definitions(
            ${target_name} PUBLIC
            USE_VULKAN_OFFLOAD
            RLLM_VULKAN_KERNEL_ROOT="${RLLM_VULKAN_KERNEL_DIR}"
        )
        target_include_directories(${target_name} PUBLIC
            ${RLLM_VULKAN_KERNEL_DIR}
            "${CMAKE_SOURCE_DIR}/kernel_compiler/runtime/include"
        )
        target_link_libraries(${target_name} PUBLIC Vulkan::Vulkan)
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

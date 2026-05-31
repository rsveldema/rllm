# ── Parallel backend selection ────────────────────────────────────────────────
# cmake -DPARALLEL_BACKEND=openmp|fastfork|sequential  (default: fastfork)
set(PARALLEL_BACKEND "fastfork" CACHE STRING "Parallel backend: openmp, fastfork, or sequential")
set_property(CACHE PARALLEL_BACKEND PROPERTY STRINGS openmp fastfork sequential)
set(OFFLOAD_BACKEND "none" CACHE STRING "Offload backend: none, hip, or vulkan")
set_property(CACHE OFFLOAD_BACKEND PROPERTY STRINGS none hip vulkan)
set(HIP_ROOT "/opt/rocm" CACHE PATH "HIP installation root")

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
    if(NOT PARALLEL_BACKEND STREQUAL "openmp")
        message(FATAL_ERROR "OFFLOAD_BACKEND=hip requires PARALLEL_BACKEND=openmp")
    endif()
    if(NOT EXISTS "${HIP_ROOT}/include")
        message(FATAL_ERROR "OFFLOAD_BACKEND=hip requires HIP_ROOT/include (set HIP_ROOT, current: '${HIP_ROOT}')")
    endif()
    message(STATUS "HIP offload macros enabled (USE_HIP_OFFLOAD)")
    message(STATUS "Using HIP root: ${HIP_ROOT}")
elseif(OFFLOAD_BACKEND STREQUAL "vulkan")
    find_package(Vulkan REQUIRED)
    message(STATUS "Vulkan offload mode enabled (USE_VULKAN_OFFLOAD)")
    message(STATUS "Vulkan includes: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan libraries: ${Vulkan_LIBRARIES}")
endif()

set(RLLM_COMPILE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(OFFLOAD_BACKEND STREQUAL "hip")
    set(RLLM_HIPIFIED_SRC_DIR "${CMAKE_BINARY_DIR}/generated/hipified/src")
    set(RLLM_HIPIFY_MANIFEST "${CMAKE_BINARY_DIR}/generated/hipified/hipified_sources.cmake")
    file(MAKE_DIRECTORY "${RLLM_HIPIFIED_SRC_DIR}")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/hipify.py"
                --src-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --out-dir "${RLLM_HIPIFIED_SRC_DIR}"
                --manifest "${RLLM_HIPIFY_MANIFEST}"
        RESULT_VARIABLE RLLM_HIPIFY_RESULT
        COMMAND_ECHO STDOUT
    )

    if(NOT RLLM_HIPIFY_RESULT EQUAL 0)
        message(FATAL_ERROR "hipify.py failed with exit code ${RLLM_HIPIFY_RESULT}")
    endif()

    set(RLLM_COMPILE_SRC_DIR "${RLLM_HIPIFIED_SRC_DIR}")
endif()

if(OFFLOAD_BACKEND STREQUAL "vulkan")
    set(RLLM_VULKANIZED_SRC_DIR "${CMAKE_BINARY_DIR}/generated/vulkanized/src")
    set(RLLM_VULKANIZE_MANIFEST "${CMAKE_BINARY_DIR}/generated/vulkanized/vulkanized_sources.cmake")
    file(MAKE_DIRECTORY "${RLLM_VULKANIZED_SRC_DIR}")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/vulkanize.py"
                --src-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --out-dir "${RLLM_VULKANIZED_SRC_DIR}"
                --manifest "${RLLM_VULKANIZE_MANIFEST}"
        RESULT_VARIABLE RLLM_VULKANIZE_RESULT
        COMMAND_ECHO STDOUT
    )

    if(NOT RLLM_VULKANIZE_RESULT EQUAL 0)
        message(FATAL_ERROR "vulkanize.py failed with exit code ${RLLM_VULKANIZE_RESULT}")
    endif()

    set(RLLM_COMPILE_SRC_DIR "${RLLM_VULKANIZED_SRC_DIR}")
endif()

# ── Helper to create an rllm static library variant ──────────────────────────
# Usage: rllm_define_lib(<target> <extra compile flags…>)
function(rllm_define_lib target_name)
    add_library(${target_name} STATIC)

    target_sources(${target_name} PRIVATE
        "${RLLM_COMPILE_SRC_DIR}/Trainer.cc"
        "${RLLM_COMPILE_SRC_DIR}/Prompter.cc"
        "${RLLM_COMPILE_SRC_DIR}/Corpus.cpp"
        "${RLLM_COMPILE_SRC_DIR}/InputLayer.cc"
        "${RLLM_COMPILE_SRC_DIR}/TransformerBlock.cc"
        "${RLLM_COMPILE_SRC_DIR}/OutputLayer.cc"
        "${RLLM_COMPILE_SRC_DIR}/NeuralNetwork.cc"
        "${RLLM_COMPILE_SRC_DIR}/serialization.cc"
        "${RLLM_COMPILE_SRC_DIR}/parallel.cc"
        "${CMAKE_BINARY_DIR}/generated/tokenizer_map.cc"
    )

    target_include_directories(${target_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
    )

    # Warning flags common to all variants; optimization flags come from ARGN.
    target_compile_options(${target_name} PRIVATE
        -Wall -Wextra -Wpedantic -Wstack-usage=10000 -fopenmp-simd -Wno-narrowing
        ${ARGN}
    )
    #target_compile_definitions(${target_name} PUBLIC RLLM_ENABLE_OVERFLOW_CHECK_ADD)

    add_dependencies(${target_name} generate_tokenizer_map)
    target_link_libraries(${target_name} PUBLIC nlohmann_json::nlohmann_json)

    if(PARALLEL_BACKEND STREQUAL "fastfork")
        target_sources(${target_name} PRIVATE "${RLLM_COMPILE_SRC_DIR}/fastfork/fastfork.cc")
        target_compile_definitions(${target_name} PUBLIC USE_FASTFORK)
        target_link_libraries(${target_name} PUBLIC hwloc)
    elseif(PARALLEL_BACKEND STREQUAL "openmp")
        target_compile_definitions(${target_name} PUBLIC USE_OPENMP)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    if(OFFLOAD_BACKEND STREQUAL "hip")
        target_compile_definitions(${target_name} PUBLIC USE_HIP_OFFLOAD)
        target_include_directories(${target_name} PUBLIC ${HIP_ROOT}/include)
    elseif(OFFLOAD_BACKEND STREQUAL "vulkan")
        target_compile_definitions(${target_name} PUBLIC USE_VULKAN_OFFLOAD)
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

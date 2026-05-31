# ── Parallel backend selection ────────────────────────────────────────────────
# cmake -DPARALLEL_BACKEND=openmp|fastfork|sequential  (default: fastfork)
set(PARALLEL_BACKEND "fastfork" CACHE STRING "Parallel backend: openmp, fastfork, or sequential")
set_property(CACHE PARALLEL_BACKEND PROPERTY STRINGS openmp fastfork sequential)
set(OFFLOAD_BACKEND "none" CACHE STRING "Offload backend: none, rocm, or vulkan")
set_property(CACHE OFFLOAD_BACKEND PROPERTY STRINGS none rocm vulkan)
set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm installation root")

if(NOT PARALLEL_BACKEND MATCHES "^(openmp|fastfork|sequential)$")
    message(FATAL_ERROR "PARALLEL_BACKEND must be one of: openmp, fastfork, sequential (got '${PARALLEL_BACKEND}')")
endif()

if(NOT OFFLOAD_BACKEND MATCHES "^(none|rocm|vulkan)$")
    message(FATAL_ERROR "OFFLOAD_BACKEND must be one of: none, rocm, vulkan (got '${OFFLOAD_BACKEND}')")
endif()

if(PARALLEL_BACKEND STREQUAL "fastfork")
    message(STATUS "Parallel backend: fastfork")
elseif(PARALLEL_BACKEND STREQUAL "openmp")
    find_package(OpenMP REQUIRED)
    message(STATUS "Parallel backend: OpenMP")
else()
    message(STATUS "Parallel backend: sequential (single-threaded)")
endif()

if(OFFLOAD_BACKEND STREQUAL "rocm")
    if(NOT PARALLEL_BACKEND STREQUAL "openmp")
        message(FATAL_ERROR "OFFLOAD_BACKEND=rocm requires PARALLEL_BACKEND=openmp")
    endif()
    if(NOT EXISTS "${ROCM_ROOT}/include")
        message(FATAL_ERROR "OFFLOAD_BACKEND=rocm requires ROCM_ROOT/include (set ROCM_ROOT, current: '${ROCM_ROOT}')")
    endif()
    message(STATUS "ROCm offload macros enabled (USE_ROCM_OFFLOAD)")
    message(STATUS "Using ROCm root: ${ROCM_ROOT}")
elseif(OFFLOAD_BACKEND STREQUAL "vulkan")
    message(STATUS "Vulkan offload mode enabled (USE_VULKAN_OFFLOAD)")
endif()

# ── Helper to create an rllm static library variant ──────────────────────────
# Usage: rllm_define_lib(<target> <extra compile flags…>)
function(rllm_define_lib target_name)
    add_library(${target_name} STATIC)

    target_sources(${target_name} PRIVATE
        Trainer.cc
        Prompter.cc
        Corpus.cpp
        InputLayer.cc
        TransformerBlock.cc
        OutputLayer.cc
        NeuralNetwork.cc
        serialization.cc
        parallel.cc
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
        target_sources(${target_name} PRIVATE fastfork/fastfork.cc)
        target_compile_definitions(${target_name} PUBLIC USE_FASTFORK)
        target_link_libraries(${target_name} PUBLIC hwloc)
    elseif(PARALLEL_BACKEND STREQUAL "openmp")
        target_compile_definitions(${target_name} PUBLIC USE_OPENMP)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    if(OFFLOAD_BACKEND STREQUAL "rocm")
        target_compile_definitions(${target_name} PUBLIC USE_ROCM_OFFLOAD)
        target_include_directories(${target_name} PUBLIC ${ROCM_ROOT}/include)
    elseif(OFFLOAD_BACKEND STREQUAL "vulkan")
        target_compile_definitions(${target_name} PUBLIC USE_VULKAN_OFFLOAD)
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

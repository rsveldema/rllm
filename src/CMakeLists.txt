add_library(rllm_lib STATIC)

# ── Parallel backend selection ────────────────────────────────────────────────
# cmake -DPARALLEL_BACKEND=openmp|fastfork|sequential  (default: openmp)
set(PARALLEL_BACKEND "fastfork" CACHE STRING "Parallel backend: openmp, fastfork, or sequential")
set_property(CACHE PARALLEL_BACKEND PROPERTY STRINGS openmp fastfork sequential)

if(NOT PARALLEL_BACKEND MATCHES "^(openmp|fastfork|sequential)$")
    message(FATAL_ERROR "PARALLEL_BACKEND must be one of: openmp, fastfork, sequential (got '${PARALLEL_BACKEND}')")
endif()

target_sources(rllm_lib PRIVATE
    RLLM.cc
    Corpus.cpp
    InputLayer.cc
    TransformerBlock.cc
    OutputLayer.cc
    NeuralNetwork.cc
    serialization.cc
    parallel.cc
    "${CMAKE_BINARY_DIR}/generated/tokenizer_map.cc"
)

target_include_directories(rllm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/generated
)

target_compile_options(rllm_lib PRIVATE -Wall -Wextra -Wpedantic -march=native -O3 -Wstack-usage=10000 -fopenmp-simd -Wno-narrowing)
#target_compile_definitions(rllm_lib PUBLIC RLLM_ENABLE_OVERFLOW_CHECK_ADD)
add_dependencies(rllm_lib generate_tokenizer_map)
target_link_libraries(rllm_lib PUBLIC nlohmann_json::nlohmann_json)

if(PARALLEL_BACKEND STREQUAL "fastfork")
    target_sources(rllm_lib PRIVATE fastfork/fastfork.cc)
    target_compile_definitions(rllm_lib PUBLIC USE_FASTFORK)
    target_link_libraries(rllm_lib PUBLIC hwloc)
    message(STATUS "Parallel backend: fastfork")
elseif(PARALLEL_BACKEND STREQUAL "openmp")
    find_package(OpenMP REQUIRED)
    target_compile_definitions(rllm_lib PUBLIC USE_OPENMP)
    target_link_libraries(rllm_lib PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "Parallel backend: OpenMP")
else()
    message(STATUS "Parallel backend: sequential (single-threaded)")
endif()



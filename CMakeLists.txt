cmake_minimum_required(VERSION 3.20)
project(rllm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckIPOSupported)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enable link-time optimization for Release builds when supported.
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    message(STATUS "LTO/IPO enabled for Release builds")
else()
    message(WARNING "LTO/IPO not supported: ${IPO_ERROR}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# --- tokenizer_map codegen ---------------------------------------------------
# Glob the input files so CMake re-runs if any of them change.
file(GLOB PREDEFINED_WORDS_FILES "${CMAKE_SOURCE_DIR}/predefined_words/*")
file(GLOB CORPUS_FILES            "${CMAKE_SOURCE_DIR}/corpus/*")
file(GLOB TOKENIZER_TRAINING_FILES "${CMAKE_SOURCE_DIR}/tokenizer_training_data/*")

set(TOKENIZER_GEN_DIR "${CMAKE_BINARY_DIR}/generated")
set(TOKENIZER_CC  "${TOKENIZER_GEN_DIR}/tokenizer_map.cc")
set(TOKENIZER_HPP "${TOKENIZER_GEN_DIR}/tokenizer_map.hpp")

file(MAKE_DIRECTORY "${TOKENIZER_GEN_DIR}")

add_custom_command(
    OUTPUT  "${TOKENIZER_CC}" "${TOKENIZER_HPP}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/create_tokenizer_map.py"
            "--cc-out"  "${TOKENIZER_CC}"
            "--hpp-out" "${TOKENIZER_HPP}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    DEPENDS
        "${CMAKE_SOURCE_DIR}/create_tokenizer_map.py"
        ${PREDEFINED_WORDS_FILES}
        ${CORPUS_FILES}
        ${TOKENIZER_TRAINING_FILES}
    COMMENT "Generating tokenizer_map.cc and tokenizer_map.hpp in build/generated"
    VERBATIM
)
add_custom_target(generate_tokenizer_map DEPENDS "${TOKENIZER_CC}" "${TOKENIZER_HPP}")
# -----------------------------------------------------------------------------

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

add_subdirectory(src)
add_subdirectory(tests)

add_executable(rllm src/main.cc
  include/LayerPrimitives.hpp
  include/Range.hpp
  include/enum_iterator.hpp
  include/fixed_size_vector.hpp
  include/fixed_size_matrix.hpp
  include/matmul.hpp
  src/fastfork/fastfork.cc
  include/fastfork/circular_queue.hpp)
target_link_libraries(rllm PRIVATE rllm_lib)
target_include_directories(rllm PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(rllm PRIVATE -O3 -march=native)

# --- Profiling target --------------------------------------------------------
add_executable(rllm_prof src/main.cc include/LayerPrimitives.hpp)
target_link_libraries(rllm_prof PRIVATE rllm_lib)
target_include_directories(rllm_prof PRIVATE ${CMAKE_SOURCE_DIR}/src)
# -O2 keeps inlining manageable; -g + no-omit-frame-pointer gives perf good stacks
target_compile_options(rllm_prof PRIVATE -O2 -g -fno-omit-frame-pointer)

set(PROFILE_ARGS "--train" "--filter" "simple" "--method" "window:2" "--epochs" "5"
    CACHE STRING "Arguments passed to rllm_prof during profiling")

add_custom_target(profile
    COMMAND perf record -F 99 --call-graph fp -o ${CMAKE_BINARY_DIR}/perf.data
            -- $<TARGET_FILE:rllm_prof> ${PROFILE_ARGS}
    COMMAND perf report --stdio --no-pager -n --percent-limit 1
            -i ${CMAKE_BINARY_DIR}/perf.data
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS rllm_prof
    COMMENT "Profiling rllm_prof (perf record + perf report). Override args via -DPROFILE_ARGS=..."
    VERBATIM
)
# -----------------------------------------------------------------------------

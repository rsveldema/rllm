cmake_minimum_required(VERSION 3.20)
project(rllm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# --- tokenizer_map codegen ---------------------------------------------------
# Glob the input files so CMake re-runs if any of them change.
file(GLOB PREDEFINED_WORDS_FILES "${CMAKE_SOURCE_DIR}/predefined_words/*")
file(GLOB CORPUS_FILES            "${CMAKE_SOURCE_DIR}/corpus/*")

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
  include/LayerPrimitives.hpp)
target_link_libraries(rllm PRIVATE rllm_lib)
target_include_directories(rllm PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(rllm PRIVATE -O3 -march=native)

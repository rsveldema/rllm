include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(RLLM_TEST_OFFLOAD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test_offload.cc")
set(RLLM_TEST_OFFLOAD_PREPROCESS_TARGET "")

if(OFFLOAD_BACKEND STREQUAL "vulkan")
    set(RLLM_VULKANIZED_TEST_DIR "${CMAKE_BINARY_DIR}/generated/vulkanized/tests")
    set(RLLM_VULKANIZED_TEST_MANIFEST "${CMAKE_BINARY_DIR}/generated/vulkanized/vulkanized_test_sources.cmake")
    set(RLLM_TEST_OFFLOAD_PREPROCESS_TARGET rllm_generate_vulkanized_test_sources)
    set(RLLM_TEST_OFFLOAD_SRC "${RLLM_VULKANIZED_TEST_DIR}/test_offload.cc")

    find_program(RLLM_TEST_VULKAN_SHADER_COMPILER NAMES glslc glslangValidator)
    if(NOT RLLM_TEST_VULKAN_SHADER_COMPILER)
        message(FATAL_ERROR "OFFLOAD_BACKEND=vulkan requires glslc or glslangValidator to compile generated test kernels to SPIR-V.")
    endif()

    add_custom_target(${RLLM_TEST_OFFLOAD_PREPROCESS_TARGET}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${RLLM_VULKANIZED_TEST_DIR}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/vulkanize.py"
                --src-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --out-dir "${RLLM_VULKANIZED_TEST_DIR}"
                --shader-compiler "${RLLM_TEST_VULKAN_SHADER_COMPILER}"
                --enum-value-tool "$<TARGET_FILE:generate_domain_replace_json>"
                --manifest "${RLLM_VULKANIZED_TEST_MANIFEST}"
        DEPENDS
            generate_domain_replace_json
            "${CMAKE_CURRENT_SOURCE_DIR}/test_offload.cc"
            "${CMAKE_SOURCE_DIR}/vulkanize.py"
            "${CMAKE_SOURCE_DIR}/vulkan-config.json"
            "${CMAKE_SOURCE_DIR}/offloadize_common.py"
        COMMENT "Generating Vulkan-offload test source tree from tests/ and compiling shaders"
        VERBATIM
    )
endif()

add_executable(test_rllm
    test_rllm.cc
    test_outputlayer.cc
    test_enum_iterator.cc
    test_enum_iterator2D.cc
    test_packed_batch.cc
    test_serialization.cc
)
target_link_libraries(test_rllm PRIVATE rllm_lib GTest::gtest_main)
target_compile_options(test_rllm PRIVATE -march=native -O3)
if(PARALLEL_BACKEND STREQUAL "openmp")
    find_package(OpenMP REQUIRED)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(test_rllm PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(test_rllm PRIVATE -fopenmp)
endif()
target_include_directories(test_rllm PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(test_rllm     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_offload
    ${RLLM_TEST_OFFLOAD_SRC}
)
set_source_files_properties(${RLLM_TEST_OFFLOAD_SRC} PROPERTIES GENERATED TRUE)
if(RLLM_TEST_OFFLOAD_PREPROCESS_TARGET)
    add_dependencies(test_offload ${RLLM_TEST_OFFLOAD_PREPROCESS_TARGET})
endif()
target_link_libraries(test_offload PRIVATE rllm_lib GTest::gtest_main)
target_compile_options(test_offload PRIVATE -march=native -O3)
if(PARALLEL_BACKEND STREQUAL "openmp")
    find_package(OpenMP REQUIRED)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(test_offload PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(test_offload PRIVATE -fopenmp)
endif()
target_include_directories(test_offload PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_offload WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

if(PARALLEL_BACKEND STREQUAL "fastfork")
    add_executable(test_fastfork
        test_fastfork.cc
    )
    target_link_libraries(test_fastfork PRIVATE rllm_lib GTest::gtest_main)
    target_compile_options(test_fastfork PRIVATE -march=native -O3)
    target_include_directories(test_fastfork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    gtest_discover_tests(test_fastfork WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

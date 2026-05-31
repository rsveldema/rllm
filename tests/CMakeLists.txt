include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_rllm
    test_rllm.cc
    test_enum_iterator.cc
    test_enum_iterator2D.cc
)
target_link_libraries(test_rllm PRIVATE rllm_lib_debug GTest::gtest_main)
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

if(PARALLEL_BACKEND STREQUAL "fastfork")
    add_executable(test_fastfork
        test_fastfork.cc
    )
    target_link_libraries(test_fastfork PRIVATE rllm_lib_debug GTest::gtest_main)
    target_compile_options(test_fastfork PRIVATE -march=native -O3)
    target_include_directories(test_fastfork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    gtest_discover_tests(test_fastfork WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

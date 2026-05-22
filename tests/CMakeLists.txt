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
)
target_link_libraries(test_rllm PRIVATE rllm_lib GTest::gtest_main -fopenmp)
target_compile_options(test_rllm PRIVATE -fopenmp -march=native -O3)
target_include_directories(test_rllm PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(test_rllm WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

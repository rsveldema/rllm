add_library(rllm_lib STATIC)

target_sources(rllm_lib PRIVATE
    RLLM.cc
)

target_include_directories(rllm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)



add_library(rllm_lib STATIC)

target_sources(rllm_lib PRIVATE
    RLLM.cc
    Corpus.cpp
)

target_include_directories(rllm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(rllm_lib PUBLIC nlohmann_json::nlohmann_json)



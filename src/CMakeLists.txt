add_library(rllm_lib STATIC)

target_sources(rllm_lib PRIVATE
    RLLM.cc
    Corpus.cpp
    InputLayer.cc
    IntermediateLayer.cc
    TransformerBlock.cc
    OutputLayer.cc
    NeuralNetwork.cc
    serialization.cc
    "${CMAKE_BINARY_DIR}/generated/tokenizer_map.cc"
)

target_include_directories(rllm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/generated
)

target_compile_options(rllm_lib PRIVATE -Wall -Wextra -Wpedantic -fopenmp -march=native -O3)
#target_compile_options(rllm_lib PRIVATE -Wall -Wextra -Wpedantic -fopenmp)
add_dependencies(rllm_lib generate_tokenizer_map)
target_link_libraries(rllm_lib PUBLIC nlohmann_json::nlohmann_json -fopenmp)



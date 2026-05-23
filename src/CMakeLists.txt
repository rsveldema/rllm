add_library(rllm_lib STATIC)

target_sources(rllm_lib PRIVATE
    RLLM.cc
    Corpus.cpp
    InputLayer.cc
    IntermediateLayer.cc
    OutputLayer.cc
    NeuralNetwork.cc
    serialization.cc
    tokenizer_map.cpp
)

target_include_directories(rllm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(rllm_lib PRIVATE -Wall -Wextra -Wpedantic -fopenmp -march=native -O3)
#target_compile_options(rllm_lib PRIVATE -Wall -Wextra -Wpedantic -fopenmp)
target_link_libraries(rllm_lib PUBLIC nlohmann_json::nlohmann_json -fopenmp)



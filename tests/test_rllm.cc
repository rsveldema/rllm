#include <gtest/gtest.h>

#include <Corpus.hpp>
#include <IntermediateLayer.hpp>

#include <chrono>
#include <cstdlib>

#include <omp.h>

TEST(PredictorTest, Placeholder) {
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Helpers shared by the OpenMP overhead benchmarks
// ---------------------------------------------------------------------------
namespace
{
    constexpr int BENCH_ITERS = 1000;
} // namespace

// ---------------------------------------------------------------------------
// Forward propagation: OpenMP overhead test
//
// Runs IntermediateLayer::propagate_forward BENCH_ITERS times with a single
// thread (serial baseline), then again with all available threads (parallel).
// Verifies that the parallel run is faster, i.e. that OMP overhead is
// outweighed by the speedup on this workload.
//
// Thread safety: forward_neuron uses std::atomic_ref CAS to scatter-accumulate
// into next_layer.m_inputs, so concurrent writes to the same target neuron
// are safe.
// ---------------------------------------------------------------------------
TEST(IntermediateLayerOpenMP, ForwardPropagationParallelFasterThanSerial)
{
    if (omp_get_max_threads() < 2)
        GTEST_SKIP() << "OpenMP thread count < 2 – no parallelism available";

    const int max_threads = omp_get_max_threads();

    std::srand(0);
    rllm::Corpus corpus;
    rllm::IntermediateLayer src(corpus);
    rllm::IntermediateLayer dst(corpus);

    std::srand(1);
    src.set_random_weights_and_connections();
    src.fill_inputs(0.5f); // ~half of neurons fire given random triggers in (0, 1)

    // --- serial baseline (1 thread) ---
    omp_set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
        src.propagate_forward(dst);
    const auto t1 = std::chrono::steady_clock::now();

    // --- OpenMP parallel ---
    omp_set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
        src.propagate_forward(dst);
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup   = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us",   serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup",     speedup);

    fprintf(stderr, "Forward \u2013 Serial: %lld\u00b5s, Parallel: %lld\u00b5s, Speedup: %.2fx\n",
            static_cast<long long>(serial_us), static_cast<long long>(parallel_us), speedup);

    EXPECT_GT(speedup, 1.0)
        << "OpenMP parallelisation of forward propagation is slower than serial "
        << "(serial=" << serial_us << "µs, parallel=" << parallel_us << "µs, speedup=" << speedup << ").\n"
        << "Consider removing or guarding the #pragma omp parallel for in "
           "IntermediateLayer::propagate_forward.";
}

// ---------------------------------------------------------------------------
// Backward propagation: OpenMP overhead test
//
// The backward loop is embarrassingly parallel: each neuron i writes only to
// its own weights[i], triggers[i], last_weight_delta[i], and prev_delta[i].
// No atomic operations are required.  Two identically-initialised layers are
// used so both runs start from the same state.
// ---------------------------------------------------------------------------
TEST(IntermediateLayerOpenMP, BackwardPropagationParallelFasterThanSerial)
{
    if (omp_get_max_threads() < 2)
        GTEST_SKIP() << "OpenMP thread count < 2 – no parallelism available";

    const int max_threads = omp_get_max_threads();

    std::srand(0);
    rllm::Corpus corpus;

    std::srand(2);
    rllm::IntermediateLayer serial_layer(corpus);
    serial_layer.set_random_weights_and_connections();
    serial_layer.fill_inputs(0.5f);

    std::srand(2); // same seed → identical state
    rllm::IntermediateLayer parallel_layer(corpus);
    parallel_layer.set_random_weights_and_connections();
    parallel_layer.fill_inputs(0.5f);

    rllm::template_token_vector<float, rllm::IntermediateLayerIndex> delta;
    delta.fill(0.5f);

    rllm::template_token_vector<float, rllm::IntermediateLayerIndex> prev_delta;

    constexpr float learning_rate = 0.01f;

    // --- serial baseline (1 thread) ---
    omp_set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        prev_delta.fill(0.0f);
        serial_layer.propagate_backward(delta, prev_delta, learning_rate);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- OpenMP parallel ---
    omp_set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        prev_delta.fill(0.0f);
        parallel_layer.propagate_backward(delta, prev_delta, learning_rate);
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup   = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us",   serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup",     speedup);

    fprintf(stderr, "Backward \u2013 Serial: %lld\u00b5s, Parallel: %lld\u00b5s, Speedup: %.2fx\n",
            static_cast<long long>(serial_us), static_cast<long long>(parallel_us), speedup);

    EXPECT_GT(speedup, 1.0)
        << "OpenMP parallelisation of backward propagation is slower than serial "
        << "(serial=" << serial_us << "µs, parallel=" << parallel_us << "µs, speedup=" << speedup << ").\n"
        << "Consider removing or guarding the #pragma omp parallel for in "
           "IntermediateLayer::propagate_backward.";
}

// ---------------------------------------------------------------------------
// Backward-from-output propagation: OpenMP overhead test
//
// Mirrors BackwardPropagationParallelFasterThanSerial but exercises
// propagate_backward_from_output_layer, where delta is indexed by TokenID
// and connections are mapped through corpus size modulo.
// Each neuron writes only to its own indices so no atomics are needed.
// ---------------------------------------------------------------------------
TEST(IntermediateLayerOpenMP, BackwardFromOutputParallelFasterThanSerial)
{
    if (omp_get_max_threads() < 2)
        GTEST_SKIP() << "OpenMP thread count < 2 \u2013 no parallelism available";

    const int max_threads = omp_get_max_threads();

    std::srand(0);
    rllm::Corpus corpus;

    std::srand(3);
    rllm::IntermediateLayer serial_layer(corpus);
    serial_layer.set_random_weights_and_connections_to_output_layer();
    serial_layer.fill_inputs(0.5f);

    std::srand(3); // same seed \u2192 identical state
    rllm::IntermediateLayer parallel_layer(corpus);
    parallel_layer.set_random_weights_and_connections_to_output_layer();
    parallel_layer.fill_inputs(0.5f);

    rllm::template_token_vector<float, rllm::TokenID> delta;
    delta.fill(0.5f);

    rllm::template_token_vector<float, rllm::IntermediateLayerIndex> prev_delta;

    constexpr float learning_rate = 0.01f;

    // --- serial baseline (1 thread) ---
    omp_set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        prev_delta.fill(0.0f);
        serial_layer.propagate_backward_from_output_layer(delta, prev_delta, learning_rate);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // --- OpenMP parallel ---
    omp_set_num_threads(max_threads);
    const auto t2 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < BENCH_ITERS; ++iter)
    {
        prev_delta.fill(0.0f);
        parallel_layer.propagate_backward_from_output_layer(delta, prev_delta, learning_rate);
    }
    const auto t3 = std::chrono::steady_clock::now();

    const auto serial_us   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto parallel_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    const double speedup   = static_cast<double>(serial_us) / static_cast<double>(parallel_us);

    RecordProperty("serial_us",   serial_us);
    RecordProperty("parallel_us", parallel_us);
    RecordProperty("speedup",     speedup);

    fprintf(stderr, "BackwardFromOutput \u2013 Serial: %lld\u00b5s, Parallel: %lld\u00b5s, Speedup: %.2fx\n",
            static_cast<long long>(serial_us), static_cast<long long>(parallel_us), speedup);

    EXPECT_GT(speedup, 1.0)
        << "OpenMP parallelisation of backward-from-output propagation is slower than serial "
        << "(serial=" << serial_us << "\u00b5s, parallel=" << parallel_us << "\u00b5s, speedup=" << speedup << ").\n"
        << "Consider removing or guarding the #pragma omp parallel for in "
           "IntermediateLayer::propagate_backward_from_output_layer.";}
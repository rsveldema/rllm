#include <LayerPrimitives.hpp>
#include <InputLayer.hpp>
#include <TransformerBlock.hpp>
#include <gtest/gtest.h>

using namespace rllm;

TEST(PackedBatchInputTest, PacksRowsAndIsolatesCausalAttention)
{
    CpuInputLine first;
    first.push_back(static_cast<TokenID>(1));
    first.push_back(static_cast<TokenID>(2));
    CpuInputLine second;
    second.push_back(static_cast<TokenID>(3));
    second.push_back(static_cast<TokenID>(4));
    second.push_back(static_cast<TokenID>(5));

    PackedBatchInput packed({first, second});

    EXPECT_EQ(static_cast<size_t>(packed.batch_size()), 2u);
    EXPECT_EQ(static_cast<size_t>(packed.packed_rows()), 5u);
    EXPECT_EQ(static_cast<size_t>(packed.row_begin(static_cast<BatchIndex>(0))), 0u);
    EXPECT_EQ(static_cast<size_t>(packed.last_row(static_cast<BatchIndex>(0))), 1u);
    EXPECT_EQ(static_cast<size_t>(packed.row_begin(static_cast<BatchIndex>(1))), 2u);
    EXPECT_EQ(static_cast<size_t>(packed.last_row(static_cast<BatchIndex>(1))), 4u);
    EXPECT_EQ(static_cast<size_t>(packed.local_position(static_cast<PositionIndex>(2))), 0u);

    EXPECT_TRUE(packed.may_attend(static_cast<PositionIndex>(4), static_cast<PositionIndex>(2)));
    EXPECT_FALSE(packed.may_attend(static_cast<PositionIndex>(2), static_cast<PositionIndex>(1)));
    EXPECT_FALSE(packed.may_attend(static_cast<PositionIndex>(1), static_cast<PositionIndex>(2)));
    EXPECT_FALSE(packed.may_attend(static_cast<PositionIndex>(3), static_cast<PositionIndex>(4)));
}

TEST(PackedBatchInputTest, BatchedTransformerForwardMatchesIndependentForwards)
{
    CpuInputLine first;
    first.push_back(static_cast<TokenID>(1));
    first.push_back(static_cast<TokenID>(2));
    CpuInputLine second;
    second.push_back(static_cast<TokenID>(3));
    second.push_back(static_cast<TokenID>(4));
    second.push_back(static_cast<TokenID>(5));

    InputLayer input_layer;
    input_layer.set_random_embeddings();
    TransformerBlock block;
    block.randomize();

    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> first_h(first.size());
    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> second_h(second.size());
    input_layer.propagate_forward(first, first_h);
    input_layer.propagate_forward(second, second_h);
    ForwardWorkspace first_ws(first.size());
    ForwardWorkspace second_ws(second.size());
    block.forward(first_h, first.size(), first_ws);
    block.forward(second_h, second.size(), second_ws);

    PackedBatchInput packed({first, second});
    GpuPackedBatchInput gpu_packed;
    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> packed_h(packed.packed_rows());
    input_layer.propagate_forward(packed, gpu_packed, packed_h);
    ForwardWorkspace packed_ws(packed.packed_rows());
    block.forward_batched(packed_h, packed.packed_rows(), gpu_packed, packed_ws);

    auto& queue = rllm::vulkan_runtime::get_queue(0);
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> first_cpu, second_cpu, packed_cpu;
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> first_q, second_q, packed_q;
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> first_attn, second_attn, packed_attn;
    cpu_fixed_triangular_matrix<float, PositionIndex, PositionIndex> first_weights, packed_weights;
    first_h.copy_to_cpu(queue, first_cpu);
    second_h.copy_to_cpu(queue, second_cpu);
    packed_h.copy_to_cpu(queue, packed_cpu);
    first_ws.Q.copy_to_cpu(queue, first_q);
    second_ws.Q.copy_to_cpu(queue, second_q);
    packed_ws.Q.copy_to_cpu(queue, packed_q);
    first_ws.attn_concat.copy_to_cpu(queue, first_attn);
    second_ws.attn_concat.copy_to_cpu(queue, second_attn);
    packed_ws.attn_concat.copy_to_cpu(queue, packed_attn);
    first_ws.attn_w[HeadsIndex::START].copy_to_cpu(queue, first_weights);
    packed_ws.attn_w[HeadsIndex::START].copy_to_cpu(queue, packed_weights);

    float max_q_difference = 0.f;
    float max_attention_difference = 0.f;
    float max_output_difference = 0.f;
    for (const auto d : enum_iterator1D<EmbeddingDimension>())
    {
        max_q_difference = std::max(max_q_difference, std::abs((packed_q[static_cast<PositionIndex>(0), d]) - (first_q[static_cast<PositionIndex>(0), d])));
        max_attention_difference = std::max(max_attention_difference, std::abs((packed_attn[static_cast<PositionIndex>(0), d]) - (first_attn[static_cast<PositionIndex>(0), d])));
        max_output_difference = std::max(max_output_difference, std::abs((packed_cpu[static_cast<PositionIndex>(0), d]) - (first_cpu[static_cast<PositionIndex>(0), d])));
        max_output_difference = std::max(max_output_difference, std::abs((packed_cpu[static_cast<PositionIndex>(1), d]) - (first_cpu[static_cast<PositionIndex>(1), d])));
        max_output_difference = std::max(max_output_difference, std::abs((packed_cpu[static_cast<PositionIndex>(2), d]) - (second_cpu[static_cast<PositionIndex>(0), d])));
        max_output_difference = std::max(max_output_difference, std::abs((packed_cpu[static_cast<PositionIndex>(4), d]) - (second_cpu[static_cast<PositionIndex>(2), d])));
    }
    EXPECT_LT(max_q_difference, 1e-4f);
    EXPECT_NEAR(
        (packed_weights[static_cast<PositionIndex>(0), static_cast<PositionIndex>(0)]),
        (first_weights[static_cast<PositionIndex>(0), static_cast<PositionIndex>(0)]),
        1e-4f
    );
    EXPECT_LT(max_attention_difference, 1e-4f);
    EXPECT_LT(max_output_difference, 1e-4f);

    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> first_dout_cpu(first.size());
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> second_dout_cpu(second.size());
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> packed_dout_cpu(packed.packed_rows());
    for (const auto d : enum_iterator1D<EmbeddingDimension>())
    {
        for (const auto row : enum_iterator1D<PositionIndex>(first.size()))
        {
            const float value = static_cast<float>((static_cast<size_t>(row) + 1) * (static_cast<size_t>(d) + 1)) / 65536.0f;
            first_dout_cpu.set(row, d, value);
            packed_dout_cpu.set(row, d, value);
        }
        for (const auto row : enum_iterator1D<PositionIndex>(second.size()))
        {
            const float value = static_cast<float>((static_cast<size_t>(row) + 3) * (static_cast<size_t>(d) + 1)) / 65536.0f;
            second_dout_cpu.set(row, d, value);
            packed_dout_cpu.set(static_cast<PositionIndex>(static_cast<size_t>(row) + static_cast<size_t>(first.size())), d, value);
        }
    }
    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> first_dout(first.size()), second_dout(second.size()), packed_dout(packed.packed_rows());
    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> first_din(first.size()), second_din(second.size()), packed_din(packed.packed_rows());
    first_dout.copy_from_cpu(queue, first_dout_cpu);
    second_dout.copy_from_cpu(queue, second_dout_cpu);
    packed_dout.copy_from_cpu(queue, packed_dout_cpu);
    BackwardWorkspace first_backward(first.size()), second_backward(second.size()), packed_backward(packed.packed_rows());
    block.backward(first_dout, first_din, first_backward, first_ws);
    block.backward(second_dout, second_din, second_backward, second_ws);
    block.backward(packed_dout, packed_din, packed_backward, packed_ws);
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> first_din_cpu, second_din_cpu, packed_din_cpu;
    first_din.copy_to_cpu(queue, first_din_cpu);
    second_din.copy_to_cpu(queue, second_din_cpu);
    packed_din.copy_to_cpu(queue, packed_din_cpu);
    float max_backward_difference = 0.f;
    for (const auto d : enum_iterator1D<EmbeddingDimension>())
    {
        for (const auto row : enum_iterator1D<PositionIndex>(first.size()))
            max_backward_difference = std::max(max_backward_difference, std::abs(packed_din_cpu[row, d] - first_din_cpu[row, d]));
        for (const auto row : enum_iterator1D<PositionIndex>(second.size()))
            max_backward_difference = std::max(max_backward_difference, std::abs(packed_din_cpu[static_cast<PositionIndex>(static_cast<size_t>(row) + static_cast<size_t>(first.size())), d] - second_din_cpu[row, d]));
    }
    EXPECT_LT(max_backward_difference, 3e-3f);
}

TEST(PackedBatchInputTest, BatchedEmbeddingMatchesIndependentForwards)
{
    CpuInputLine first;
    first.push_back(static_cast<TokenID>(1));
    first.push_back(static_cast<TokenID>(2));
    CpuInputLine second;
    second.push_back(static_cast<TokenID>(3));
    second.push_back(static_cast<TokenID>(4));
    second.push_back(static_cast<TokenID>(5));

    InputLayer layer;
    layer.set_random_embeddings();
    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> first_h(first.size());
    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> second_h(second.size());
    layer.propagate_forward(first, first_h);
    layer.propagate_forward(second, second_h);

    PackedBatchInput packed({first, second});
    GpuPackedBatchInput gpu_packed;
    flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> packed_h(packed.packed_rows());
    layer.propagate_forward(packed, gpu_packed, packed_h);

    auto& queue = rllm::vulkan_runtime::get_queue(0);
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> first_cpu;
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> second_cpu;
    cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> packed_cpu;
    first_h.copy_to_cpu(queue, first_cpu);
    second_h.copy_to_cpu(queue, second_cpu);
    packed_h.copy_to_cpu(queue, packed_cpu);

    for (const auto d : enum_iterator1D<EmbeddingDimension>())
    {
        EXPECT_FLOAT_EQ((packed_cpu[static_cast<PositionIndex>(0), d]), (first_cpu[static_cast<PositionIndex>(0), d]));
        EXPECT_FLOAT_EQ((packed_cpu[static_cast<PositionIndex>(1), d]), (first_cpu[static_cast<PositionIndex>(1), d]));
        EXPECT_FLOAT_EQ((packed_cpu[static_cast<PositionIndex>(2), d]), (second_cpu[static_cast<PositionIndex>(0), d]));
        EXPECT_FLOAT_EQ((packed_cpu[static_cast<PositionIndex>(4), d]), (second_cpu[static_cast<PositionIndex>(2), d]));
    }
}

#include <ck_fmha.hpp>

#include <cmath>
#include <cstdio>

#include <fmha_fwd.hpp>
#include <hip_session.hpp>
#include <LayerPrimitives.hpp>

namespace rllm
{
    bool ck_fmha_forward(HipQueue& queue,
        const VBaseDeviceBuffer& q,
        const VBaseDeviceBuffer& k,
        const VBaseDeviceBuffer& v,
        VBaseDeviceBuffer& output,
        VBaseDeviceBuffer& lse,
        const VBaseDeviceBuffer& sequence_starts,
        int packed_rows,
        int batch_size,
        int max_sequence_length)
    {
        constexpr int head_dimension = 64;
        constexpr int head_count = static_cast<int>(EmbeddingDimension::MAX) / head_dimension;
        constexpr int embedding_dimension = static_cast<int>(EmbeddingDimension::MAX);

        fmha_fwd_traits traits{};
        traits.hdim_q = head_dimension;
        traits.hdim_v = head_dimension;
        traits.data_type = "fp16";
        traits.is_group_mode = true;
        traits.is_v_rowmajor = true;
        traits.has_logits_soft_cap = false;
        traits.mask_type = mask_enum::mask_top_left;
        traits.bias_type = bias_enum::no_bias;
        traits.has_lse = true;
        traits.has_dropout = false;
        traits.qscale_type = quant_scale_enum::no_scale;

        fmha_fwd_args args{};
        args.q_ptr = q.get();
        args.k_ptr = k.get();
        args.v_ptr = v.get();
        args.o_ptr = output.get();
        args.lse_ptr = lse.get();
        args.seqstart_q_ptr = sequence_starts.get();
        args.seqstart_k_ptr = sequence_starts.get();
        args.seqlen_q = packed_rows;
        args.seqlen_k = packed_rows;
        args.batch = batch_size;
        args.max_seqlen_q = max_sequence_length;
        args.hdim_q = head_dimension;
        args.hdim_v = head_dimension;
        args.nhead_q = head_count;
        args.nhead_k = head_count;
        args.num_head_q_total = head_count;
        args.scale_s = 1.0f / std::sqrt(static_cast<float>(head_dimension));
        args.stride_q = embedding_dimension;
        args.stride_k = embedding_dimension;
        args.stride_v = embedding_dimension;
        args.stride_o = embedding_dimension;
        args.nhead_stride_q = head_dimension;
        args.nhead_stride_k = head_dimension;
        args.nhead_stride_v = head_dimension;
        args.nhead_stride_o = head_dimension;
        args.nhead_stride_lse = packed_rows;
        args.batch_stride_q = 0;
        args.batch_stride_k = 0;
        args.batch_stride_v = 0;
        args.batch_stride_o = 0;
        args.batch_stride_lse = 0;
        args.window_size_left = -1;
        args.window_size_right = 0;
        args.mask_type = static_cast<int>(mask_enum::mask_top_left);
        args.p_drop = 0.0f;
        args.s_randval = false;
        args.drop_seed_offset = std::make_pair(uint64_t{0}, uint64_t{0});

        const ck_tile::stream_config stream_config{queue.stream(), false};
        const float elapsed = fmha_fwd(traits, args, stream_config);
        if(elapsed < 0.0f)
        {
            std::fprintf(stderr, "CK FMHA found no compatible forward kernel\n");
            return false;
        }
        return true;
    }
}

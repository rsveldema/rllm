#pragma once

#include <cstddef>

class HipQueue;
class VBaseDeviceBuffer;

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
        int max_sequence_length);
}

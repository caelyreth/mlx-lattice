#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

[[kernel]]
void precise_feature_projection_f32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& in_channels [[buffer(4)]],
    constant const int& out_channels [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    const int total = rows * out_channels;
    if (int(index) >= total) {
        return;
    }
    const int row = int(index) / out_channels;
    const int output = int(index) % out_channels;
    float accumulator = 0.0F;
    for (int input = 0; input < in_channels; ++input) {
        accumulator += feats[row * in_channels + input] *
                       weights[output * in_channels + input];
    }
    out[index] = accumulator;
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void precise_feature_projection_f32_c32(
    device float* feats [[buffer(0)]],
    device float* weights [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& in_channels [[buffer(4)]],
    constant const int& out_channels [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int tile_rows = 64;
    constexpr int channels = 32;
    const int row_start = int(group_id) * tile_rows;

    if (row_start + tile_rows <= rows) {
        threadgroup float rhs_tile[channels * channels];
        for (int index = int(tid); index < channels * channels; index += 128) {
            const int input = index / channels;
            const int output = index % channels;
            rhs_tile[index] = weights[output * channels + input];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr auto descriptor = matmul2d_descriptor(
            tile_rows,
            channels,
            channels,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate
        );
        matmul2d<descriptor, execution_simdgroups<4>> op;
        auto lhs = tensor(
            feats + row_start * channels,
            extents<int32_t, channels, tile_rows>(),
            array<int32_t, 2>{1, channels}
        );
        auto rhs = tensor(
            rhs_tile,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        );
        auto destination = op.get_destination_cooperative_tensor<
            decltype(lhs),
            decltype(rhs),
            float>();
#pragma unroll
        for (uint16_t index = 0; index < destination.get_capacity(); ++index) {
            if (destination.is_valid_element(index)) {
                destination[index] = 0.0F;
            }
        }
        op.run(lhs, rhs, destination);
        auto out_tensor = tensor(
            out + row_start * channels,
            extents<int32_t, channels, tile_rows>(),
            array<int32_t, 2>{1, channels}
        );
        destination.store(out_tensor);
        return;
    }

    for (int index = int(tid); index < tile_rows * channels; index += 128) {
        const int row = row_start + index / channels;
        const int output = index % channels;
        if (row >= rows) {
            continue;
        }
        float accumulator = 0.0F;
        for (int input = 0; input < channels; ++input) {
            accumulator += feats[row * channels + input] *
                           weights[output * channels + input];
        }
        out[row * channels + output] = accumulator;
    }

    (void)in_channels;
    (void)out_channels;
}

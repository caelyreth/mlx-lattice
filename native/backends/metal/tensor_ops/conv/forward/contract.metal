#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

#include "native/backends/metal/conv/common.metal"

namespace {

enum : int {
    kTileRows = 16,
    kTileChannels = 16,
    kTileValues = 16 * 16,
    kKernelVolume = 27,
};

template <typename T>
inline void load_row_block(
    device const T* feats,
    device const int* in_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* row_offsets,
    int edge_capacity,
    int out_capacity,
    int out_row_base,
    int kernel_id,
    int ci_base,
    int feat_s0,
    int feat_s1,
    threadgroup float* lhs_tile,
    uint lane
) {
    const int edge_count = min(counts[0], edge_capacity);
    const int out_count = min(counts[1], out_capacity);
    for (int row_slot = 0; row_slot < kTileRows; ++row_slot) {
        const int out_row = out_row_base + row_slot;
        if (out_row >= out_count) {
            continue;
        }
        const int begin = max(row_offsets[out_row], 0);
        const int end = min(row_offsets[out_row + 1], edge_count);
        int in_row = -1;
        for (int edge = begin; edge < end; ++edge) {
            if (kernel_ids[edge] != kernel_id) {
                continue;
            }
            in_row = in_rows[edge];
            break;
        }
        if (in_row < 0) {
            continue;
        }
        const int feat_base = in_row * feat_s0;
        for (uint ci = lane; ci < kTileChannels; ci += 32) {
            lhs_tile[int(ci) * kTileRows + row_slot] =
                float(feats[feat_base + (ci_base + int(ci)) * feat_s1]);
        }
    }
}

template <typename T>
inline void load_weight_block(
    device const T* weights,
    int kernel_id,
    int ci_base,
    int co_base,
    int weight_s0,
    int weight_s1,
    int weight_s2,
    int weight_s3,
    int weight_s4,
    int weight_layout,
    int kernel_x,
    int kernel_y,
    int kernel_z,
    threadgroup float* rhs_tile,
    uint lane
) {
    for (uint index = lane; index < kTileValues; index += 32) {
        const int co = int(index) / kTileChannels;
        const int ci = int(index) - co * kTileChannels;
        rhs_tile[int(index)] = float(weights[sparse_conv_weight_offset(
            kernel_id,
            ci_base + ci,
            co_base + co,
            weight_layout,
            kernel_x,
            kernel_y,
            kernel_z,
            weight_s0,
            weight_s1,
            weight_s2,
            weight_s3,
            weight_s4
        )]);
    }
}

template <typename T>
inline void store_output_block(
    device T* out,
    int out_channels,
    int out_row_base,
    int co_base,
    int out_count,
    threadgroup float* out_tile,
    uint lane
) {
    for (uint index = lane; index < kTileValues; index += 32) {
        const int row = int(index) / kTileChannels;
        const int co = int(index) - row * kTileChannels;
        const int out_row = out_row_base + row;
        if (out_row < out_count) {
            out[out_row * out_channels + co_base + co] =
                T(out_tile[int(index)]);
        }
    }
}

template <typename T>
inline void sparse_relation_conv_forward_implicit_gemm_impl(
    device const T* feats,
    device const T* weights,
    device const int* in_rows,
    device const int* out_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* row_offsets,
    device T* out,
    constant const int& edge_capacity,
    constant const int& out_capacity,
    constant const int& n_kernels,
    constant const int& in_channels,
    constant const int& out_channels,
    constant const int& feat_s0,
    constant const int& feat_s1,
    constant const int& weight_s0,
    constant const int& weight_s1,
    constant const int& weight_s2,
    constant const int& weight_s3,
    constant const int& weight_s4,
    constant const int& weight_layout,
    constant const int& kernel_x,
    constant const int& kernel_y,
    constant const int& kernel_z,
    threadgroup float* lhs_tile,
    threadgroup float* rhs_tile,
    threadgroup float* out_tile,
    uint group_id,
    uint lane
) {
    const int co_blocks = out_channels / kTileChannels;
    const int row_tile = int(group_id) / co_blocks;
    const int co_block = int(group_id) % co_blocks;
    const int out_row_base = row_tile * kTileRows;
    const int co_base = co_block * kTileChannels;
    const int out_count = min(counts[1], out_capacity);
    if (out_row_base >= out_count) {
        return;
    }

    for (uint index = lane; index < kTileValues; index += 32) {
        out_tile[int(index)] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = matmul2d_descriptor(
        kTileRows,
        kTileChannels,
        kTileChannels,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroup> op;
    auto lhs_tensor =
        tensor<threadgroup float, extents<int32_t, 16, 16>, tensor_inline>(
            lhs_tile, extents<int32_t, 16, 16>()
        );
    auto rhs_tensor =
        tensor<threadgroup float, extents<int32_t, 16, 16>, tensor_inline>(
            rhs_tile, extents<int32_t, 16, 16>()
        );
    auto out_tensor =
        tensor<threadgroup float, extents<int32_t, 16, 16>, tensor_inline>(
            out_tile, extents<int32_t, 16, 16>()
        );

    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        for (int ci_base = 0; ci_base < in_channels; ci_base += kTileChannels) {
            for (uint index = lane; index < kTileValues; index += 32) {
                lhs_tile[int(index)] = 0.0f;
                rhs_tile[int(index)] = 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            load_row_block(
                feats,
                in_rows,
                kernel_ids,
                counts,
                row_offsets,
                edge_capacity,
                out_capacity,
                out_row_base,
                kernel_id,
                ci_base,
                feat_s0,
                feat_s1,
                lhs_tile,
                lane
            );
            load_weight_block(
                weights,
                kernel_id,
                ci_base,
                co_base,
                weight_s0,
                weight_s1,
                weight_s2,
                weight_s3,
                weight_s4,
                weight_layout,
                kernel_x,
                kernel_y,
                kernel_z,
                rhs_tile,
                lane
            );
            simdgroup_barrier(mem_flags::mem_threadgroup);
            op.run(lhs_tensor, rhs_tensor, out_tensor);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    store_output_block(
        out, out_channels, out_row_base, co_base, out_count, out_tile, lane
    );
    (void)out_rows;
}

} // namespace

[[kernel, max_total_threads_per_threadgroup(32)]] void
sparse_relation_conv_forward_implicit_gemm_f16_i32_c64(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& in_channels [[buffer(11)]],
    constant const int& out_channels [[buffer(12)]],
    constant const int& feat_s0 [[buffer(13)]],
    constant const int& feat_s1 [[buffer(14)]],
    constant const int& weight_s0 [[buffer(15)]],
    constant const int& weight_s1 [[buffer(16)]],
    constant const int& weight_s2 [[buffer(17)]],
    constant const int& weight_s3 [[buffer(18)]],
    constant const int& weight_s4 [[buffer(19)]],
    constant const int& weight_layout [[buffer(20)]],
    constant const int& kernel_x [[buffer(21)]],
    constant const int& kernel_y [[buffer(22)]],
    constant const int& kernel_z [[buffer(23)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float lhs_tile[kTileValues];
    threadgroup float rhs_tile[kTileValues];
    threadgroup float out_tile[kTileValues];
    sparse_relation_conv_forward_implicit_gemm_impl(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        edge_capacity,
        out_capacity,
        n_kernels,
        in_channels,
        out_channels,
        feat_s0,
        feat_s1,
        weight_s0,
        weight_s1,
        weight_s2,
        weight_s3,
        weight_s4,
        weight_layout,
        kernel_x,
        kernel_y,
        kernel_z,
        lhs_tile,
        rhs_tile,
        out_tile,
        group_id,
        lane
    );
}

[[kernel, max_total_threads_per_threadgroup(32)]] void
sparse_relation_conv_forward_implicit_gemm_f32_i32_c64(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& in_channels [[buffer(11)]],
    constant const int& out_channels [[buffer(12)]],
    constant const int& feat_s0 [[buffer(13)]],
    constant const int& feat_s1 [[buffer(14)]],
    constant const int& weight_s0 [[buffer(15)]],
    constant const int& weight_s1 [[buffer(16)]],
    constant const int& weight_s2 [[buffer(17)]],
    constant const int& weight_s3 [[buffer(18)]],
    constant const int& weight_s4 [[buffer(19)]],
    constant const int& weight_layout [[buffer(20)]],
    constant const int& kernel_x [[buffer(21)]],
    constant const int& kernel_y [[buffer(22)]],
    constant const int& kernel_z [[buffer(23)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float lhs_tile[kTileValues];
    threadgroup float rhs_tile[kTileValues];
    threadgroup float out_tile[kTileValues];
    sparse_relation_conv_forward_implicit_gemm_impl(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        edge_capacity,
        out_capacity,
        n_kernels,
        in_channels,
        out_channels,
        feat_s0,
        feat_s1,
        weight_s0,
        weight_s1,
        weight_s2,
        weight_s3,
        weight_s4,
        weight_layout,
        kernel_x,
        kernel_y,
        kernel_z,
        lhs_tile,
        rhs_tile,
        out_tile,
        group_id,
        lane
    );
}

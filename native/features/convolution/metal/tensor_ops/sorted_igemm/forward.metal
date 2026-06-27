#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

#include "native/features/convolution/metal/common.metal"

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

[[kernel, max_total_threads_per_threadgroup(128)]]
void row_stationary_tensor_coop_devicew_rowfill4x2_kvmap_f16acc_sorted_c32_m64(
    device const half* feats [[buffer(0)]],
    device half* weights_kci_co [[buffer(1)]],
    device const int* sorted_kv_out_in_map [[buffer(2)]],
    device const int* reorder_rows [[buffer(3)]],
    device const int* tile_masks [[buffer(4)]],
    device half* out [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& store_sorted [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int tile_rows = 64;
    constexpr int channels = 32;
    constexpr int chunks_per_row = channels / 4;

    threadgroup half lhs_tile[tile_rows * channels];

    const int row_start = int(group_id) * tile_rows;
    const int mask_base = int(group_id) * 4;
    uint active_mask = uint(
        tile_masks[mask_base + 0] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );
    const uint original_mask = active_mask;

    constexpr auto desc = matmul2d_descriptor(
        tile_rows,
        channels,
        channels,
        false,
        false,
        true,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroups<4>> op;
    auto lhs_tensor = tensor<
        threadgroup half,
        extents<int32_t, channels, tile_rows>,
        tensor_inline>(lhs_tile, extents<int32_t, channels, tile_rows>());
    auto out_tensor = op.get_destination_cooperative_tensor<
        decltype(lhs_tensor),
        decltype(tensor(
            weights_kci_co,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        )),
        half>();

#pragma unroll
    for (uint16_t i = 0; i < out_tensor.get_capacity(); ++i) {
        if (out_tensor.is_valid_element(i)) {
            out_tensor[i] = half(0.0h);
        }
    }

    while (active_mask != 0) {
        const int kv = ctz(active_mask);
        active_mask &= active_mask - 1;
        const int kv_base = kv * rows;

        if (tid < tile_rows * 2) {
            const int row_slot = int(tid >> 1);
            const int half_slot = int(tid & 1);
            const int sorted_row = row_start + row_slot;
            const int in_row = sorted_row < rows
                                   ? sorted_kv_out_in_map[kv_base + sorted_row]
                                   : -1;
            const int tile_base = row_slot * channels + half_slot * 16;
            if (in_row >= 0) {
                const int feat_base = in_row * channels + half_slot * 16;
#pragma unroll
                for (int chunk = 0; chunk < chunks_per_row / 2; ++chunk) {
                    const int ci = chunk * 4;
                    *reinterpret_cast<threadgroup half4*>(
                        lhs_tile + tile_base + ci
                    ) =
                        *reinterpret_cast<device const half4*>(
                            feats + feat_base + ci
                        );
                }
            } else {
#pragma unroll
                for (int chunk = 0; chunk < chunks_per_row / 2; ++chunk) {
                    const int ci = chunk * 4;
                    *reinterpret_cast<threadgroup half4*>(
                        lhs_tile + tile_base + ci
                    ) = half4(0.0h);
                }
            }
        }

        auto rhs_tensor = tensor(
            weights_kci_co + kv * channels * channels,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(lhs_tensor, rhs_tensor, out_tensor);
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (store_sorted != 0 && row_start + tile_rows <= rows) {
        auto device_out = tensor(
            out + row_start * channels,
            extents<int32_t, channels, tile_rows>(),
            array<int32_t, 2>{1, channels}
        );
        out_tensor.store(device_out);
    } else {
#pragma unroll
        for (uint16_t i = 0; i < out_tensor.get_capacity(); ++i) {
            if (!out_tensor.is_valid_element(i)) {
                continue;
            }
            const auto idx = out_tensor.get_multidimensional_index(i);
            const int co = int(idx[0]);
            const int row_slot = int(idx[1]);
            const int sorted_row = row_start + row_slot;
            if (sorted_row < rows) {
                const int out_row =
                    store_sorted != 0 ? sorted_row : reorder_rows[sorted_row];
                out[out_row * channels + co] =
                    original_mask == 0 ? half(0.0h) : out_tensor[i];
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void row_stationary_tensor_coop_devicew_full64_kvmap_f16acc_sorted_c64_m64(
    device const half* feats [[buffer(0)]],
    device half* weights_kci_co [[buffer(1)]],
    device const int* sorted_kv_out_in_map [[buffer(2)]],
    device const int* reorder_rows [[buffer(3)]],
    device const int* tile_masks [[buffer(4)]],
    device half* out [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& store_sorted [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int tile_rows = 64;
    constexpr int channels = 64;
    constexpr int chunks_per_row = channels / 4;

    threadgroup half lhs_tile[tile_rows * channels];

    const int row_start = int(group_id) * tile_rows;
    const int mask_base = int(group_id) * 4;
    uint active_mask = uint(
        tile_masks[mask_base + 0] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );
    const uint original_mask = active_mask;

    constexpr auto desc = matmul2d_descriptor(
        tile_rows,
        channels,
        channels,
        false,
        false,
        true,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroups<4>> op;
    auto lhs_tensor = tensor<
        threadgroup half,
        extents<int32_t, channels, tile_rows>,
        tensor_inline>(lhs_tile, extents<int32_t, channels, tile_rows>());
    auto out_tensor = op.get_destination_cooperative_tensor<
        decltype(lhs_tensor),
        decltype(tensor(
            weights_kci_co,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        )),
        half>();

#pragma unroll
    for (uint16_t i = 0; i < out_tensor.get_capacity(); ++i) {
        if (out_tensor.is_valid_element(i)) {
            out_tensor[i] = half(0.0h);
        }
    }

    while (active_mask != 0) {
        const int kv = ctz(active_mask);
        active_mask &= active_mask - 1;
        const int kv_base = kv * rows;

        if (tid < tile_rows * 2) {
            const int row_slot = int(tid >> 1);
            const int half_slot = int(tid & 1);
            const int sorted_row = row_start + row_slot;
            const int in_row = sorted_row < rows
                                   ? sorted_kv_out_in_map[kv_base + sorted_row]
                                   : -1;
            const int tile_base = row_slot * channels + half_slot * 32;
            if (in_row >= 0) {
                const int feat_base = in_row * channels + half_slot * 32;
#pragma unroll
                for (int chunk = 0; chunk < chunks_per_row / 2; ++chunk) {
                    const int ci = chunk * 4;
                    *reinterpret_cast<threadgroup half4*>(
                        lhs_tile + tile_base + ci
                    ) =
                        *reinterpret_cast<device const half4*>(
                            feats + feat_base + ci
                        );
                }
            } else {
#pragma unroll
                for (int chunk = 0; chunk < chunks_per_row / 2; ++chunk) {
                    const int ci = chunk * 4;
                    *reinterpret_cast<threadgroup half4*>(
                        lhs_tile + tile_base + ci
                    ) = half4(0.0h);
                }
            }
        }

        auto rhs_tensor = tensor(
            weights_kci_co + kv * channels * channels,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(lhs_tensor, rhs_tensor, out_tensor);
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (store_sorted != 0 && row_start + tile_rows <= rows) {
        auto device_out = tensor(
            out + row_start * channels,
            extents<int32_t, channels, tile_rows>(),
            array<int32_t, 2>{1, channels}
        );
        out_tensor.store(device_out);
    } else {
#pragma unroll
        for (uint16_t i = 0; i < out_tensor.get_capacity(); ++i) {
            if (!out_tensor.is_valid_element(i)) {
                continue;
            }
            const auto idx = out_tensor.get_multidimensional_index(i);
            const int co = int(idx[0]);
            const int row_slot = int(idx[1]);
            const int sorted_row = row_start + row_slot;
            if (sorted_row < rows) {
                const int out_row =
                    store_sorted != 0 ? sorted_row : reorder_rows[sorted_row];
                out[out_row * channels + co] =
                    original_mask == 0 ? half(0.0h) : out_tensor[i];
            }
        }
    }
}

#include <metal_stdlib>

using namespace metal;

#include "native/features/convolution/metal/common.metal"

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

static inline float4
load_packed_weight4(device const half* weights_kci_co, int kv, int ci, int co) {
    constexpr int channels = 32;
    const int offset = kv * channels * channels + ci * channels + co;
    return float4(
        *reinterpret_cast<device const half4*>(weights_kci_co + offset)
    );
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void row_stationary_direct_packedw_c32_m64(
    device const half* feats [[buffer(0)]],
    device half* weights_kci_co [[buffer(1)]],
    device const int* sorted_out_in_map [[buffer(2)]],
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

    const int row_slot = int(tid) / 2;
    const int co = (int(tid) - row_slot * 2) * 16;
    const int sorted_row = int(group_id) * tile_rows + row_slot;
    if (sorted_row >= rows) {
        return;
    }

    const int mask_base = int(group_id) * 4;
    uint active_mask = uint(
        tile_masks[mask_base + 0] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    while (active_mask != 0) {
        const int kv = ctz(active_mask);
        active_mask &= active_mask - 1;
        const int in_row = sorted_out_in_map[sorted_row * 27 + kv];
        if (in_row < 0) {
            continue;
        }
        const int feat_base = in_row * channels;
        for (int ci = 0; ci < channels; ++ci) {
            const float feat = float(feats[feat_base + ci]);
            acc0 += feat * load_packed_weight4(weights_kci_co, kv, ci, co + 0);
            acc1 += feat * load_packed_weight4(weights_kci_co, kv, ci, co + 4);
            acc2 += feat * load_packed_weight4(weights_kci_co, kv, ci, co + 8);
            acc3 += feat * load_packed_weight4(weights_kci_co, kv, ci, co + 12);
        }
    }

    const int out_row =
        store_sorted != 0 ? sorted_row : reorder_rows[sorted_row];
    const int out_base = out_row * channels + co;
    *reinterpret_cast<device half4*>(out + out_base + 0) = half4(acc0);
    *reinterpret_cast<device half4*>(out + out_base + 4) = half4(acc1);
    *reinterpret_cast<device half4*>(out + out_base + 8) = half4(acc2);
    *reinterpret_cast<device half4*>(out + out_base + 12) = half4(acc3);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void row_stationary_direct_packedw_c64_m64(
    device const half* feats [[buffer(0)]],
    device half* weights_kci_co [[buffer(1)]],
    device const int* sorted_out_in_map [[buffer(2)]],
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

    const int linear = int(tid);
    const int row_slot = linear / 2;
    const int co_base = (linear - row_slot * 2) * 32;
    const int sorted_row = int(group_id) * tile_rows + row_slot;
    if (sorted_row >= rows) {
        return;
    }

    const int mask_base = int(group_id) * 4;
    uint active_mask = uint(
        tile_masks[mask_base + 0] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );
    float acc[32];
    for (int index = 0; index < 32; ++index) {
        acc[index] = 0.0f;
    }

    while (active_mask != 0) {
        const int kv = ctz(active_mask);
        active_mask &= active_mask - 1;
        const int in_row = sorted_out_in_map[sorted_row * 27 + kv];
        if (in_row < 0) {
            continue;
        }
        for (int ci = 0; ci < channels; ++ci) {
            const float feat = float(feats[in_row * channels + ci]);
            const int weight_base =
                kv * channels * channels + ci * channels + co_base;
            for (int co = 0; co < 32; ++co) {
                acc[co] += feat * float(weights_kci_co[weight_base + co]);
            }
        }
    }

    const int out_row =
        store_sorted != 0 ? sorted_row : reorder_rows[sorted_row];
    const int out_base = out_row * channels + co_base;
    for (int co = 0; co < 32; co += 4) {
        *reinterpret_cast<device half4*>(out + out_base + co) =
            half4(acc[co + 0], acc[co + 1], acc[co + 2], acc[co + 3]);
    }
}

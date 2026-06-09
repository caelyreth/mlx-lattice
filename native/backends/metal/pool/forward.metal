#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/pool/common.metal"

[[kernel]] void sparse_pool_clear_f32_i32(
    device int* out_coords [[buffer(0)]],
    device float* out_feats [[buffer(1)]],
    device int* counts [[buffer(2)]],
    constant const int& coord_total [[buffer(3)]],
    constant const int& feat_total [[buffer(4)]],
    constant const int& reduce [[buffer(5)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem == 0) {
        counts[0] = 0;
        counts[1] = 0;
    }
    if (elem < uint(coord_total)) {
        out_coords[elem] = 0;
    }
    if (elem < uint(feat_total)) {
        out_feats[elem] = reduce == 1 ? -INFINITY : 0.0f;
    }
}

[[kernel]] void sparse_pool_forward_gather_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const int* offsets [[buffer(3)]],
    device const int* out_coords [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device float* out_feats [[buffer(6)]],
    constant const int& reduce [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& feat_s0 [[buffer(18)]],
    constant const int& feat_s1 [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = n_out_rows * channels;
    if (elem >= uint(total)) {
        return;
    }
    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    int rows = min(active_rows[0], n_in_rows);
    if (out_row >= counts[1]) {
        return;
    }

    int out_base = out_row * 4;
    float acc = reduce == 1 ? -INFINITY : 0.0f;
    int degree = 0;
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int offset_base = kernel_id * 3;
        int input_coord[4] = {
            out_coords[out_base],
            out_coords[out_base + 1] * stride_x + offsets[offset_base] - pad_x,
            out_coords[out_base + 2] * stride_y + offsets[offset_base + 1] -
                pad_y,
            out_coords[out_base + 3] * stride_z + offsets[offset_base + 2] -
                pad_z,
        };
        int in_row = -1;
        if (!pool_find_input_row(coords, rows, input_coord, in_row)) {
            continue;
        }
        float value = feats[in_row * feat_s0 + channel * feat_s1];
        acc = reduce == 1 ? max(acc, value) : acc + value;
        ++degree;
    }
    out_feats[elem] = reduce == 2 ? acc / float(max(degree, 1)) : acc;
}

[[kernel]] void sparse_pool_downsample_gather_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const int* out_coords [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    device float* out_feats [[buffer(5)]],
    constant const int& reduce [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    constant const int& n_out_rows [[buffer(8)]],
    constant const int& channels [[buffer(9)]],
    constant const int& stride_x [[buffer(10)]],
    constant const int& stride_y [[buffer(11)]],
    constant const int& stride_z [[buffer(12)]],
    constant const int& feat_s0 [[buffer(13)]],
    constant const int& feat_s1 [[buffer(14)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = n_out_rows * channels;
    if (elem >= uint(total)) {
        return;
    }
    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    int rows = min(active_rows[0], n_in_rows);
    if (out_row >= counts[1]) {
        return;
    }

    int out_base = out_row * 4;
    int target[4] = {
        out_coords[out_base],
        out_coords[out_base + 1],
        out_coords[out_base + 2],
        out_coords[out_base + 3],
    };
    float acc = reduce == 1 ? -INFINITY : 0.0f;
    int degree = 0;
    for (int in_row = 0; in_row < rows; ++in_row) {
        int candidate[4];
        pool_downsample_coord(
            coords, in_row, stride_x, stride_y, stride_z, candidate
        );
        if (!pool_coord_equal(candidate, target)) {
            continue;
        }
        float value = feats[in_row * feat_s0 + channel * feat_s1];
        acc = reduce == 1 ? max(acc, value) : acc + value;
        ++degree;
    }
    out_feats[elem] = reduce == 2 ? acc / float(max(degree, 1)) : acc;
}

[[kernel]] void sparse_pool_downsample_gather_rows_f32_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const int* out_coords [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    device float* out_feats [[buffer(5)]],
    constant const int& reduce [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    constant const int& n_out_rows [[buffer(8)]],
    constant const int& channels [[buffer(9)]],
    constant const int& stride_x [[buffer(10)]],
    constant const int& stride_y [[buffer(11)]],
    constant const int& stride_z [[buffer(12)]],
    constant const int& feat_s0 [[buffer(13)]],
    constant const int& feat_s1 [[buffer(14)]],
    uint out_row [[thread_position_in_grid]]
) {
    if (out_row >= uint(n_out_rows) || out_row >= uint(counts[1])) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    int out_base = int(out_row) * 4;
    int target[4] = {
        out_coords[out_base],
        out_coords[out_base + 1],
        out_coords[out_base + 2],
        out_coords[out_base + 3],
    };
    int out_feat_base = int(out_row) * channels;
    for (int channel = 0; channel < channels; ++channel) {
        out_feats[out_feat_base + channel] = reduce == 1 ? -INFINITY : 0.0f;
    }

    int degree = 0;
    for (int in_row = 0; in_row < rows; ++in_row) {
        int candidate[4];
        pool_downsample_coord(
            coords, in_row, stride_x, stride_y, stride_z, candidate
        );
        if (!pool_coord_equal(candidate, target)) {
            continue;
        }
        for (int channel = 0; channel < channels; ++channel) {
            int out_index = out_feat_base + channel;
            float value = feats[in_row * feat_s0 + channel * feat_s1];
            out_feats[out_index] = reduce == 1
                                       ? max(out_feats[out_index], value)
                                       : out_feats[out_index] + value;
        }
        ++degree;
    }
    if (reduce == 2) {
        float scale = 1.0f / float(max(degree, 1));
        for (int channel = 0; channel < channels; ++channel) {
            out_feats[out_feat_base + channel] *= scale;
        }
    }
}

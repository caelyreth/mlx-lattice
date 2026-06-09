#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/pool/common.metal"

inline void pool_input_coord(
    device const int* out_coords,
    device const int* offsets,
    int out_row,
    int kernel_id,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    thread int* input_coord
) {
    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    input_coord[0] = out_coords[out_base];
    input_coord[1] =
        out_coords[out_base + 1] * stride_x + offsets[offset_base] - pad_x;
    input_coord[2] =
        out_coords[out_base + 2] * stride_y + offsets[offset_base + 1] - pad_y;
    input_coord[3] =
        out_coords[out_base + 3] * stride_z + offsets[offset_base + 2] - pad_z;
}

[[kernel]] void sparse_pool_autodiff_clear_f32(
    device float* out [[buffer(0)]],
    constant const int& total [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(total)) {
        out[elem] = 0.0f;
    }
}

[[kernel]] void sparse_pool_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* coords [[buffer(3)]],
    device const int* active_rows [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device const int* out_coords [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device atomic_float* grad [[buffer(8)]],
    constant const int& reduce [[buffer(9)]],
    constant const int& n_in_rows [[buffer(10)]],
    constant const int& n_out_rows [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& channels [[buffer(13)]],
    constant const int& stride_x [[buffer(14)]],
    constant const int& stride_y [[buffer(15)]],
    constant const int& stride_z [[buffer(16)]],
    constant const int& pad_x [[buffer(17)]],
    constant const int& pad_y [[buffer(18)]],
    constant const int& pad_z [[buffer(19)]],
    constant const int& cotangent_s0 [[buffer(20)]],
    constant const int& cotangent_s1 [[buffer(21)]],
    constant const int& feat_s0 [[buffer(22)]],
    constant const int& feat_s1 [[buffer(23)]],
    constant const int& pooled_s0 [[buffer(24)]],
    constant const int& pooled_s1 [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = n_out_rows * channels;
    if (elem >= uint(total)) {
        return;
    }
    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    if (out_row >= counts[1]) {
        return;
    }

    int rows = min(active_rows[0], n_in_rows);
    float pooled_value = pooled[out_row * pooled_s0 + channel * pooled_s1];
    int contributors = 0;
    if (reduce != 0) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int input_coord[4];
            pool_input_coord(
                out_coords,
                offsets,
                out_row,
                kernel_id,
                stride_x,
                stride_y,
                stride_z,
                pad_x,
                pad_y,
                pad_z,
                input_coord
            );
            int in_row = -1;
            if (!pool_find_input_row(coords, rows, input_coord, in_row)) {
                continue;
            }
            if (reduce == 2 ||
                feats[in_row * feat_s0 + channel * feat_s1] == pooled_value) {
                ++contributors;
            }
        }
    }

    float scale = reduce == 0 ? 1.0f : 1.0f / float(max(contributors, 1));
    float contribution =
        cotangent[out_row * cotangent_s0 + channel * cotangent_s1] * scale;
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int input_coord[4];
        pool_input_coord(
            out_coords,
            offsets,
            out_row,
            kernel_id,
            stride_x,
            stride_y,
            stride_z,
            pad_x,
            pad_y,
            pad_z,
            input_coord
        );
        int in_row = -1;
        if (!pool_find_input_row(coords, rows, input_coord, in_row)) {
            continue;
        }
        if (reduce == 1 &&
            feats[in_row * feat_s0 + channel * feat_s1] != pooled_value) {
            continue;
        }
        atomic_fetch_add_explicit(
            &grad[in_row * channels + channel],
            contribution,
            memory_order_relaxed
        );
    }
}

[[kernel]] void sparse_pool_jvp_f32_i32(
    device const float* tangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* coords [[buffer(3)]],
    device const int* active_rows [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device const int* out_coords [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant const int& reduce [[buffer(9)]],
    constant const int& n_in_rows [[buffer(10)]],
    constant const int& n_out_rows [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& channels [[buffer(13)]],
    constant const int& stride_x [[buffer(14)]],
    constant const int& stride_y [[buffer(15)]],
    constant const int& stride_z [[buffer(16)]],
    constant const int& pad_x [[buffer(17)]],
    constant const int& pad_y [[buffer(18)]],
    constant const int& pad_z [[buffer(19)]],
    constant const int& tangent_s0 [[buffer(20)]],
    constant const int& tangent_s1 [[buffer(21)]],
    constant const int& feat_s0 [[buffer(22)]],
    constant const int& feat_s1 [[buffer(23)]],
    constant const int& pooled_s0 [[buffer(24)]],
    constant const int& pooled_s1 [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = n_out_rows * channels;
    if (elem >= uint(total)) {
        return;
    }
    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    if (out_row >= counts[1]) {
        out[elem] = 0.0f;
        return;
    }

    int rows = min(active_rows[0], n_in_rows);
    float pooled_value = pooled[out_row * pooled_s0 + channel * pooled_s1];
    float value = 0.0f;
    int contributors = 0;
    int first_rank = n_in_rows * n_kernels;
    float first_tangent = 0.0f;
    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        int input_coord[4];
        pool_input_coord(
            out_coords,
            offsets,
            out_row,
            kernel_id,
            stride_x,
            stride_y,
            stride_z,
            pad_x,
            pad_y,
            pad_z,
            input_coord
        );
        int in_row = -1;
        if (!pool_find_input_row(coords, rows, input_coord, in_row)) {
            continue;
        }
        float tangent_value =
            tangent[in_row * tangent_s0 + channel * tangent_s1];
        if (reduce == 1) {
            if (feats[in_row * feat_s0 + channel * feat_s1] != pooled_value) {
                continue;
            }
            int rank = in_row * n_kernels + kernel_id;
            if (rank < first_rank) {
                first_rank = rank;
                first_tangent = tangent_value;
            }
            continue;
        }
        value += tangent_value;
        ++contributors;
    }

    if (reduce == 1) {
        value = first_tangent;
    } else if (reduce == 2) {
        value /= float(max(contributors, 1));
    }
    out[elem] = value;
}

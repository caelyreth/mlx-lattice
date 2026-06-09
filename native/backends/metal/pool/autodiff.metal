#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/pool/common.metal"

[[kernel]] void sparse_pool_grad_f32_i32_serial(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* coords [[buffer(3)]],
    device const int* active_rows [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device float* grad [[buffer(6)]],
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
    constant const int& cotangent_s0 [[buffer(18)]],
    constant const int& cotangent_s1 [[buffer(19)]],
    constant const int& feat_s0 [[buffer(20)]],
    constant const int& feat_s1 [[buffer(21)]],
    constant const int& pooled_s0 [[buffer(22)]],
    constant const int& pooled_s1 [[buffer(23)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    (void)n_out_rows;
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_in_rows * channels; ++index) {
        grad[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (!pool_relation(
                    coords,
                    rows,
                    kernel_id,
                    offsets,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    in_row,
                    out_coord,
                    out_row
                )) {
                continue;
            }
            int degree = pool_degree(
                coords,
                offsets,
                rows,
                n_kernels,
                out_row,
                stride_x,
                stride_y,
                stride_z,
                pad_x,
                pad_y,
                pad_z
            );
            for (int channel = 0; channel < channels; ++channel) {
                float scale = 1.0f;
                if (reduce == 1) {
                    float feat_value =
                        feats[in_row * feat_s0 + channel * feat_s1];
                    float pooled_value =
                        pooled[out_row * pooled_s0 + channel * pooled_s1];
                    if (feat_value != pooled_value) {
                        continue;
                    }
                    int tie_count = pool_max_tie_count(
                        coords,
                        offsets,
                        feats,
                        rows,
                        n_kernels,
                        out_row,
                        channel,
                        pooled_value,
                        stride_x,
                        stride_y,
                        stride_z,
                        pad_x,
                        pad_y,
                        pad_z,
                        feat_s0,
                        feat_s1
                    );
                    if (tie_count == 0) {
                        continue;
                    }
                    scale = 1.0f / float(tie_count);
                } else if (reduce == 2) {
                    scale = 1.0f / float(degree);
                }
                grad[in_row * channels + channel] +=
                    cotangent[out_row * cotangent_s0 + channel * cotangent_s1] *
                    scale;
            }
        }
    }
}

[[kernel]] void sparse_pool_jvp_f32_i32_serial(
    device const float* tangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* coords [[buffer(3)]],
    device const int* active_rows [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device float* out [[buffer(6)]],
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
    constant const int& tangent_s0 [[buffer(18)]],
    constant const int& tangent_s1 [[buffer(19)]],
    constant const int& feat_s0 [[buffer(20)]],
    constant const int& feat_s1 [[buffer(21)]],
    constant const int& pooled_s0 [[buffer(22)]],
    constant const int& pooled_s1 [[buffer(23)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_out_rows * channels; ++index) {
        out[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (!pool_relation(
                    coords,
                    rows,
                    kernel_id,
                    offsets,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    in_row,
                    out_coord,
                    out_row
                )) {
                continue;
            }
            int degree = pool_degree(
                coords,
                offsets,
                rows,
                n_kernels,
                out_row,
                stride_x,
                stride_y,
                stride_z,
                pad_x,
                pad_y,
                pad_z
            );
            for (int channel = 0; channel < channels; ++channel) {
                float scale = 1.0f;
                if (reduce == 1) {
                    float feat_value =
                        feats[in_row * feat_s0 + channel * feat_s1];
                    float pooled_value =
                        pooled[out_row * pooled_s0 + channel * pooled_s1];
                    if (feat_value != pooled_value) {
                        continue;
                    }
                    int first_rank = pool_first_max_rank(
                        coords,
                        offsets,
                        feats,
                        rows,
                        n_kernels,
                        out_row,
                        channel,
                        pooled_value,
                        stride_x,
                        stride_y,
                        stride_z,
                        pad_x,
                        pad_y,
                        pad_z,
                        feat_s0,
                        feat_s1
                    );
                    if (in_row * n_kernels + kernel_id != first_rank) {
                        continue;
                    }
                } else if (reduce == 2) {
                    scale = 1.0f / float(degree);
                }
                out[out_row * channels + channel] +=
                    tangent[in_row * tangent_s0 + channel * tangent_s1] * scale;
            }
        }
    }
}

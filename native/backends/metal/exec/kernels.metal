#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/exec/common.metal"

[[kernel]] void sparse_conv_f32_i32_serial(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    device float* out_feats [[buffer(6)]],
    device int* counts [[buffer(7)]],
    constant const int& map_op [[buffer(8)]],
    constant const int& n_in_rows [[buffer(9)]],
    constant const int& n_out_rows [[buffer(10)]],
    constant const int& n_kernels [[buffer(11)]],
    constant const int& in_channels [[buffer(12)]],
    constant const int& out_channels [[buffer(13)]],
    constant const int& stride_x [[buffer(14)]],
    constant const int& stride_y [[buffer(15)]],
    constant const int& stride_z [[buffer(16)]],
    constant const int& pad_x [[buffer(17)]],
    constant const int& pad_y [[buffer(18)]],
    constant const int& pad_z [[buffer(19)]],
    constant const int& feat_s0 [[buffer(20)]],
    constant const int& feat_s1 [[buffer(21)]],
    constant const int& weight_s0 [[buffer(22)]],
    constant const int& weight_s1 [[buffer(23)]],
    constant const int& weight_s2 [[buffer(24)]],
    constant const int& weight_s3 [[buffer(25)]],
    constant const int& weight_s4 [[buffer(26)]],
    constant const int& weight_layout [[buffer(27)]],
    constant const int& kernel_x [[buffer(28)]],
    constant const int& kernel_y [[buffer(29)]],
    constant const int& kernel_z [[buffer(30)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_out_rows * 4; ++index) {
        out_coords[index] = 0;
    }
    for (int index = 0; index < n_out_rows * out_channels; ++index) {
        out_feats[index] = 0.0f;
    }

    int out_count = 0;
    int edge_count = 0;
    if (map_op == 0) {
        for (int row = 0; row < rows; ++row) {
            int candidate[4];
            downsample_coord(
                coords, row, stride_x, stride_y, stride_z, candidate
            );
            if (!seen_forward_coord(
                    coords, row, stride_x, stride_y, stride_z, candidate
                )) {
                write_coord(out_coords, out_count, candidate);
                out_count += 1;
            }
        }
        for (int out_row = 0; out_row < out_count; ++out_row) {
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int out_base = out_row * 4;
                int input_coord[4] = {
                    out_coords[out_base],
                    out_coords[out_base + 1] * stride_x + offsets[offset_base] -
                        pad_x,
                    out_coords[out_base + 2] * stride_y +
                        offsets[offset_base + 1] - pad_y,
                    out_coords[out_base + 3] * stride_z +
                        offsets[offset_base + 2] - pad_z,
                };
                int in_row = -1;
                if (!find_input_row(coords, rows, input_coord, in_row)) {
                    continue;
                }
                edge_count += 1;
                for (int ci = 0; ci < in_channels; ++ci) {
                    float value = feats[in_row * feat_s0 + ci * feat_s1];
                    int feat_base = out_row * out_channels;
                    for (int co = 0; co < out_channels; ++co) {
                        out_feats[feat_base + co] +=
                            value * weights[weight_offset(
                                        kernel_id,
                                        ci,
                                        co,
                                        weight_layout,
                                        kernel_x,
                                        kernel_y,
                                        kernel_z,
                                        weight_s0,
                                        weight_s1,
                                        weight_s2,
                                        weight_s3,
                                        weight_s4
                                    )];
                    }
                }
            }
        }
    } else {
        for (int in_row = 0; in_row < rows; ++in_row) {
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int candidate[4];
                transposed_candidate(
                    coords,
                    offsets,
                    in_row,
                    kernel_id,
                    stride_x,
                    stride_y,
                    stride_z,
                    map_op == 2 ? 0 : pad_x,
                    map_op == 2 ? 0 : pad_y,
                    map_op == 2 ? 0 : pad_z,
                    candidate
                );
                int out_row = edge_count;
                if (map_op == 1) {
                    out_row = transposed_out_row_for_coord(
                        coords,
                        offsets,
                        rows,
                        n_kernels,
                        stride_x,
                        stride_y,
                        stride_z,
                        pad_x,
                        pad_y,
                        pad_z,
                        candidate
                    );
                }
                if (out_row == out_count) {
                    write_coord(out_coords, out_row, candidate);
                    out_count += 1;
                } else if (map_op == 2) {
                    write_coord(out_coords, out_row, candidate);
                    out_count += 1;
                }
                edge_count += 1;
                for (int ci = 0; ci < in_channels; ++ci) {
                    float value = feats[in_row * feat_s0 + ci * feat_s1];
                    int feat_base = out_row * out_channels;
                    for (int co = 0; co < out_channels; ++co) {
                        out_feats[feat_base + co] +=
                            value * weights[weight_offset(
                                        kernel_id,
                                        ci,
                                        co,
                                        weight_layout,
                                        kernel_x,
                                        kernel_y,
                                        kernel_z,
                                        weight_s0,
                                        weight_s1,
                                        weight_s2,
                                        weight_s3,
                                        weight_s4
                                    )];
                    }
                }
            }
        }
    }

    counts[0] = edge_count;
    counts[1] = out_count;
}

[[kernel]] void sparse_conv_input_grad_f32_i32_serial(
    device const float* cotangent [[buffer(0)]],
    device const int* coords [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device float* grad [[buffer(5)]],
    constant const int& map_op [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    constant const int& n_out_rows [[buffer(8)]],
    constant const int& n_kernels [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& cotangent_s0 [[buffer(18)]],
    constant const int& cotangent_s1 [[buffer(19)]],
    constant const int& weight_s0 [[buffer(20)]],
    constant const int& weight_s1 [[buffer(21)]],
    constant const int& weight_s2 [[buffer(22)]],
    constant const int& weight_s3 [[buffer(23)]],
    constant const int& weight_s4 [[buffer(24)]],
    constant const int& weight_layout [[buffer(25)]],
    constant const int& kernel_x [[buffer(26)]],
    constant const int& kernel_y [[buffer(27)]],
    constant const int& kernel_z [[buffer(28)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_in_rows * in_channels; ++index) {
        grad[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (map_op == 0) {
                if (!valid_forward_relation_coord(
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
            } else if (map_op == 2) {
                out_row = in_row * n_kernels + kernel_id;
            } else {
                transposed_candidate(
                    coords,
                    offsets,
                    in_row,
                    kernel_id,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    out_coord
                );
                out_row = transposed_out_row_for_coord(
                    coords,
                    offsets,
                    rows,
                    n_kernels,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    out_coord
                );
            }
            if (out_row < 0 || out_row >= n_out_rows) {
                continue;
            }
            for (int ci = 0; ci < in_channels; ++ci) {
                for (int co = 0; co < out_channels; ++co) {
                    grad[in_row * in_channels + ci] +=
                        cotangent[out_row * cotangent_s0 + co * cotangent_s1] *
                        weights[weight_offset(
                            kernel_id,
                            ci,
                            co,
                            weight_layout,
                            kernel_x,
                            kernel_y,
                            kernel_z,
                            weight_s0,
                            weight_s1,
                            weight_s2,
                            weight_s3,
                            weight_s4
                        )];
                }
            }
        }
    }
}

[[kernel]] void sparse_conv_weight_grad_f32_i32_serial(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* coords [[buffer(2)]],
    device const int* active_rows [[buffer(3)]],
    device const int* offsets [[buffer(4)]],
    device float* grad [[buffer(5)]],
    constant const int& map_op [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    constant const int& n_out_rows [[buffer(8)]],
    constant const int& n_kernels [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& stride_x [[buffer(12)]],
    constant const int& stride_y [[buffer(13)]],
    constant const int& stride_z [[buffer(14)]],
    constant const int& pad_x [[buffer(15)]],
    constant const int& pad_y [[buffer(16)]],
    constant const int& pad_z [[buffer(17)]],
    constant const int& feat_s0 [[buffer(18)]],
    constant const int& feat_s1 [[buffer(19)]],
    constant const int& cotangent_s0 [[buffer(20)]],
    constant const int& cotangent_s1 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_kernels * in_channels * out_channels;
         ++index) {
        grad[index] = 0.0f;
    }
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (map_op == 0) {
                if (!valid_forward_relation_coord(
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
            } else if (map_op == 2) {
                out_row = in_row * n_kernels + kernel_id;
            } else {
                transposed_candidate(
                    coords,
                    offsets,
                    in_row,
                    kernel_id,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    out_coord
                );
                out_row = transposed_out_row_for_coord(
                    coords,
                    offsets,
                    rows,
                    n_kernels,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    out_coord
                );
            }
            if (out_row < 0 || out_row >= n_out_rows) {
                continue;
            }
            for (int ci = 0; ci < in_channels; ++ci) {
                for (int co = 0; co < out_channels; ++co) {
                    grad[dense_weight_offset(
                        kernel_id,
                        ci,
                        co,
                        weight_layout,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        in_channels,
                        out_channels
                    )] += feats[in_row * feat_s0 + ci * feat_s1] *
                          cotangent[out_row * cotangent_s0 + co * cotangent_s1];
                }
            }
        }
    }
}

[[kernel]] void sparse_pool_f32_i32_serial(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const float* feats [[buffer(2)]],
    device const int* offsets [[buffer(3)]],
    device int* out_coords [[buffer(4)]],
    device float* out_feats [[buffer(5)]],
    device int* counts [[buffer(6)]],
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
    if (elem != 0) {
        return;
    }
    int rows = min(active_rows[0], n_in_rows);
    for (int index = 0; index < n_out_rows * 4; ++index) {
        out_coords[index] = 0;
    }
    for (int index = 0; index < n_out_rows * channels; ++index) {
        out_feats[index] = reduce == 1 ? -INFINITY : 0.0f;
    }

    int out_count = 0;
    for (int row = 0; row < rows; ++row) {
        int candidate[4];
        downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
        if (!seen_forward_coord(
                coords, row, stride_x, stride_y, stride_z, candidate
            )) {
            write_coord(out_coords, out_count, candidate);
            out_count += 1;
        }
    }

    int edge_count = 0;
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
            int out_coord[4];
            int out_row = -1;
            if (!valid_forward_relation_coord(
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
            edge_count += 1;
            for (int channel = 0; channel < channels; ++channel) {
                int out_index = out_row * channels + channel;
                float value = feats[in_row * feat_s0 + channel * feat_s1];
                if (reduce == 1) {
                    out_feats[out_index] = max(out_feats[out_index], value);
                } else {
                    out_feats[out_index] += value;
                }
            }
        }
    }
    if (reduce == 2) {
        for (int out_row = 0; out_row < out_count; ++out_row) {
            int degree = degree_for_forward_out_row(
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
                out_feats[out_row * channels + channel] /= float(degree);
            }
        }
    }
    counts[0] = edge_count;
    counts[1] = out_count;
}

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
            if (!valid_forward_relation_coord(
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
            int degree = degree_for_forward_out_row(
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
                int in_index = in_row * channels + channel;
                float feat_value = feats[in_row * feat_s0 + channel * feat_s1];
                float pooled_value =
                    pooled[out_row * pooled_s0 + channel * pooled_s1];
                if (reduce == 1 && feat_value != pooled_value) {
                    continue;
                }
                float scale = reduce == 2 ? 1.0f / float(degree) : 1.0f;
                grad[in_index] +=
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
            if (!valid_forward_relation_coord(
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
            int degree = degree_for_forward_out_row(
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
                float feat_value = feats[in_row * feat_s0 + channel * feat_s1];
                float pooled_value =
                    pooled[out_row * pooled_s0 + channel * pooled_s1];
                if (reduce == 1 && feat_value != pooled_value) {
                    continue;
                }
                float scale = reduce == 2 ? 1.0f / float(degree) : 1.0f;
                out[out_row * channels + channel] +=
                    tangent[in_row * tangent_s0 + channel * tangent_s1] * scale;
            }
        }
    }
}

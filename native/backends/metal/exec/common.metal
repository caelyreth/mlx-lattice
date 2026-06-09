#include "native/backends/metal/coords/common.metal"

inline bool coord_equal4(thread const int* lhs, thread const int* rhs) {
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] &&
           lhs[3] == rhs[3];
}

inline void downsample_coord(
    device const int* coords,
    int row,
    int stride_x,
    int stride_y,
    int stride_z,
    thread int* out
) {
    int base = row * 4;
    out[0] = coords[base];
    out[1] = floor_div_int(coords[base + 1], stride_x);
    out[2] = floor_div_int(coords[base + 2], stride_y);
    out[3] = floor_div_int(coords[base + 3], stride_z);
}

inline bool seen_forward_coord(
    device const int* coords,
    int row,
    int stride_x,
    int stride_y,
    int stride_z,
    thread const int* candidate
) {
    for (int prev = 0; prev < row; ++prev) {
        int previous[4];
        downsample_coord(coords, prev, stride_x, stride_y, stride_z, previous);
        if (coord_equal4(previous, candidate)) {
            return true;
        }
    }
    return false;
}

inline int forward_out_row_for_coord(
    device const int* coords,
    int rows,
    int stride_x,
    int stride_y,
    int stride_z,
    thread const int* target
) {
    int out_row = 0;
    for (int row = 0; row < rows; ++row) {
        int candidate[4];
        downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
        if (seen_forward_coord(
                coords, row, stride_x, stride_y, stride_z, candidate
            )) {
            continue;
        }
        if (coord_equal4(candidate, target)) {
            return out_row;
        }
        out_row += 1;
    }
    return -1;
}

inline bool find_input_row(
    device const int* coords,
    int rows,
    thread const int* target,
    thread int& out_row
) {
    for (int row = 0; row < rows; ++row) {
        if (coord4_equal(target, coords, row)) {
            out_row = row;
            return true;
        }
    }
    return false;
}

inline int exec_coord_hash_i32(int b, int x, int y, int z) {
    uint hash = 2166136261u;
    hash = (hash ^ uint(b)) * 16777619u;
    hash = (hash ^ uint(x)) * 16777619u;
    hash = (hash ^ uint(y)) * 16777619u;
    hash = (hash ^ uint(z)) * 16777619u;
    int out = int(hash & 0x7fffffffu);
    return out == int(0x7fffffff) ? out - 1 : out;
}

inline int exec_coord_hash_i32(device const int* coords, int row) {
    int base = row * 4;
    return exec_coord_hash_i32(
        coords[base], coords[base + 1], coords[base + 2], coords[base + 3]
    );
}

inline int exec_coord_hash_i32(thread const int* coord) {
    return exec_coord_hash_i32(coord[0], coord[1], coord[2], coord[3]);
}

inline int lookup_input_row_hash(
    device const int* coords,
    device const int* table_keys,
    device const int* table_rows,
    int table_capacity,
    int empty_key,
    thread const int* target
) {
    int key = exec_coord_hash_i32(target);
    int slot = key & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int found = table_keys[slot];
        if (found == empty_key) {
            return -1;
        }
        if (found == key) {
            int row = table_rows[slot];
            if (row >= 0 && coord4_equal(target, coords, row)) {
                return row;
            }
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
    return -1;
}

inline bool valid_forward_relation_coord(
    device const int* coords,
    int rows,
    int kernel_id,
    device const int* offsets,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    int in_row,
    thread int* out_coord,
    thread int& out_row
) {
    int in_base = in_row * 4;
    int offset_base = kernel_id * 3;
    int vx = coords[in_base + 1] - offsets[offset_base] + pad_x;
    int vy = coords[in_base + 2] - offsets[offset_base + 1] + pad_y;
    int vz = coords[in_base + 3] - offsets[offset_base + 2] + pad_z;
    if (vx % stride_x != 0 || vy % stride_y != 0 || vz % stride_z != 0) {
        return false;
    }
    out_coord[0] = coords[in_base];
    out_coord[1] = vx / stride_x;
    out_coord[2] = vy / stride_y;
    out_coord[3] = vz / stride_z;
    out_row = forward_out_row_for_coord(
        coords, rows, stride_x, stride_y, stride_z, out_coord
    );
    return out_row >= 0;
}

inline void transposed_candidate(
    device const int* coords,
    device const int* offsets,
    int in_row,
    int kernel_id,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    thread int* out
) {
    int in_base = in_row * 4;
    int offset_base = kernel_id * 3;
    out[0] = coords[in_base];
    out[1] = coords[in_base + 1] * stride_x + offsets[offset_base] - pad_x;
    out[2] = coords[in_base + 2] * stride_y + offsets[offset_base + 1] - pad_y;
    out[3] = coords[in_base + 3] * stride_z + offsets[offset_base + 2] - pad_z;
}

inline int transposed_out_row_for_coord(
    device const int* coords,
    device const int* offsets,
    int rows,
    int kernels,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    thread const int* target
) {
    int out_row = 0;
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < kernels; ++kernel_id) {
            int candidate[4];
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
                candidate
            );
            bool seen = false;
            for (int prev_in = 0; prev_in <= in_row; ++prev_in) {
                int limit = prev_in == in_row ? kernel_id : kernels;
                for (int prev_kernel = 0; prev_kernel < limit; ++prev_kernel) {
                    int previous[4];
                    transposed_candidate(
                        coords,
                        offsets,
                        prev_in,
                        prev_kernel,
                        stride_x,
                        stride_y,
                        stride_z,
                        pad_x,
                        pad_y,
                        pad_z,
                        previous
                    );
                    if (coord_equal4(previous, candidate)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) {
                    break;
                }
            }
            if (seen) {
                continue;
            }
            if (coord_equal4(candidate, target)) {
                return out_row;
            }
            out_row += 1;
        }
    }
    return -1;
}

inline int degree_for_forward_out_row(
    device const int* coords,
    device const int* offsets,
    int rows,
    int kernels,
    int out_row,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z
) {
    int degree = 0;
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < kernels; ++kernel_id) {
            int candidate[4];
            int edge_out = -1;
            if (valid_forward_relation_coord(
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
                    candidate,
                    edge_out
                ) &&
                edge_out == out_row) {
                degree += 1;
            }
        }
    }
    return max(degree, 1);
}

inline int max_pool_tie_count_for_forward_out_row_channel(
    device const int* coords,
    device const int* offsets,
    device const float* feats,
    int rows,
    int kernels,
    int out_row,
    int channel,
    float pooled_value,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    int feat_s0,
    int feat_s1
) {
    int count = 0;
    for (int probe_in = 0; probe_in < rows; ++probe_in) {
        for (int probe_kernel = 0; probe_kernel < kernels; ++probe_kernel) {
            int candidate[4];
            int probe_out = -1;
            if (valid_forward_relation_coord(
                    coords,
                    rows,
                    probe_kernel,
                    offsets,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    probe_in,
                    candidate,
                    probe_out
                ) &&
                probe_out == out_row &&
                feats[probe_in * feat_s0 + channel * feat_s1] == pooled_value) {
                count += 1;
            }
        }
    }
    return count;
}

inline int max_pool_first_rank_for_forward_out_row_channel(
    device const int* coords,
    device const int* offsets,
    device const float* feats,
    int rows,
    int kernels,
    int out_row,
    int channel,
    float pooled_value,
    int stride_x,
    int stride_y,
    int stride_z,
    int pad_x,
    int pad_y,
    int pad_z,
    int feat_s0,
    int feat_s1
) {
    for (int probe_in = 0; probe_in < rows; ++probe_in) {
        for (int probe_kernel = 0; probe_kernel < kernels; ++probe_kernel) {
            int candidate[4];
            int probe_out = -1;
            if (valid_forward_relation_coord(
                    coords,
                    rows,
                    probe_kernel,
                    offsets,
                    stride_x,
                    stride_y,
                    stride_z,
                    pad_x,
                    pad_y,
                    pad_z,
                    probe_in,
                    candidate,
                    probe_out
                ) &&
                probe_out == out_row &&
                feats[probe_in * feat_s0 + channel * feat_s1] == pooled_value) {
                return probe_in * kernels + probe_kernel;
            }
        }
    }
    return -1;
}

inline int weight_offset(
    int kernel_id,
    int in_channel,
    int out_channel,
    int weight_layout,
    int kernel_x,
    int kernel_y,
    int kernel_z,
    int weight_s0,
    int weight_s1,
    int weight_s2,
    int weight_s3,
    int weight_s4
) {
    if (weight_layout == 0) {
        return kernel_id * weight_s0 + in_channel * weight_s1 +
               out_channel * weight_s2;
    }

    int xy = kernel_y * kernel_z;
    int kx = kernel_id / xy;
    int rem = kernel_id % xy;
    int ky = rem / kernel_z;
    int kz = rem % kernel_z;
    (void)kernel_x;
    return out_channel * weight_s0 + kx * weight_s1 + ky * weight_s2 +
           kz * weight_s3 + in_channel * weight_s4;
}

inline float4 weight4_at(
    device const float* weights,
    int kernel_id,
    int in_channel,
    int out_channel,
    int weight_layout,
    int kernel_y,
    int kernel_z,
    int weight_s0,
    int weight_s1,
    int weight_s2,
    int weight_s3,
    int weight_s4
) {
    if (weight_layout == 0) {
        int base = kernel_id * weight_s0 + in_channel * weight_s1 +
                   out_channel * weight_s2;
        return float4(
            weights[base],
            weights[base + weight_s2],
            weights[base + 2 * weight_s2],
            weights[base + 3 * weight_s2]
        );
    }

    int xy = kernel_y * kernel_z;
    int kx = kernel_id / xy;
    int rem = kernel_id % xy;
    int ky = rem / kernel_z;
    int kz = rem % kernel_z;
    int base = out_channel * weight_s0 + kx * weight_s1 + ky * weight_s2 +
               kz * weight_s3 + in_channel * weight_s4;
    return float4(
        weights[base],
        weights[base + weight_s0],
        weights[base + 2 * weight_s0],
        weights[base + 3 * weight_s0]
    );
}

inline int dense_weight_offset(
    int kernel_id,
    int in_channel,
    int out_channel,
    int weight_layout,
    int kernel_x,
    int kernel_y,
    int kernel_z,
    int in_channels,
    int out_channels
) {
    if (weight_layout == 0) {
        return (kernel_id * in_channels + in_channel) * out_channels +
               out_channel;
    }

    int xy = kernel_y * kernel_z;
    int kx = kernel_id / xy;
    int rem = kernel_id % xy;
    int ky = rem / kernel_z;
    int kz = rem % kernel_z;
    return (((out_channel * kernel_x + kx) * kernel_y + ky) * kernel_z + kz) *
               in_channels +
           in_channel;
}

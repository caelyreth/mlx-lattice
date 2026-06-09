#include "native/backends/metal/coords/common.metal"

inline bool pool_coord_equal(thread const int* lhs, thread const int* rhs) {
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] &&
           lhs[3] == rhs[3];
}

inline void pool_downsample_coord(
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

inline bool pool_coord_seen(
    device const int* coords,
    int row,
    int stride_x,
    int stride_y,
    int stride_z,
    thread const int* candidate
) {
    for (int previous_row = 0; previous_row < row; ++previous_row) {
        int previous[4];
        pool_downsample_coord(
            coords, previous_row, stride_x, stride_y, stride_z, previous
        );
        if (pool_coord_equal(previous, candidate)) {
            return true;
        }
    }
    return false;
}

inline int pool_output_row(
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
        pool_downsample_coord(
            coords, row, stride_x, stride_y, stride_z, candidate
        );
        if (pool_coord_seen(
                coords, row, stride_x, stride_y, stride_z, candidate
            )) {
            continue;
        }
        if (pool_coord_equal(candidate, target)) {
            return out_row;
        }
        ++out_row;
    }
    return -1;
}

inline bool pool_find_input_row(
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

inline int pool_coord_hash(thread const int* coord) {
    uint hash = 2166136261u;
    hash = (hash ^ uint(coord[0])) * 16777619u;
    hash = (hash ^ uint(coord[1])) * 16777619u;
    hash = (hash ^ uint(coord[2])) * 16777619u;
    hash = (hash ^ uint(coord[3])) * 16777619u;
    int out = int(hash & 0x7fffffffu);
    return out == int(0x7fffffff) ? out - 1 : out;
}

inline bool pool_relation(
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
    int x = coords[in_base + 1] - offsets[offset_base] + pad_x;
    int y = coords[in_base + 2] - offsets[offset_base + 1] + pad_y;
    int z = coords[in_base + 3] - offsets[offset_base + 2] + pad_z;
    if (x % stride_x != 0 || y % stride_y != 0 || z % stride_z != 0) {
        return false;
    }
    out_coord[0] = coords[in_base];
    out_coord[1] = x / stride_x;
    out_coord[2] = y / stride_y;
    out_coord[3] = z / stride_z;
    out_row =
        pool_output_row(coords, rows, stride_x, stride_y, stride_z, out_coord);
    return out_row >= 0;
}

inline int pool_degree(
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
            int candidate_row = -1;
            if (pool_relation(
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
                    candidate_row
                ) &&
                candidate_row == out_row) {
                ++degree;
            }
        }
    }
    return max(degree, 1);
}

inline int pool_max_tie_count(
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
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < kernels; ++kernel_id) {
            int candidate[4];
            int candidate_row = -1;
            if (pool_relation(
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
                    candidate_row
                ) &&
                candidate_row == out_row &&
                feats[in_row * feat_s0 + channel * feat_s1] == pooled_value) {
                ++count;
            }
        }
    }
    return count;
}

inline int pool_first_max_rank(
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
    for (int in_row = 0; in_row < rows; ++in_row) {
        for (int kernel_id = 0; kernel_id < kernels; ++kernel_id) {
            int candidate[4];
            int candidate_row = -1;
            if (pool_relation(
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
                    candidate_row
                ) &&
                candidate_row == out_row &&
                feats[in_row * feat_s0 + channel * feat_s1] == pooled_value) {
                return in_row * kernels + kernel_id;
            }
        }
    }
    return -1;
}

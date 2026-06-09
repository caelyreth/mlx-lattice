#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/pool/common.metal"

[[kernel]] void sparse_pool_forward_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant const int& n_in_rows [[buffer(5)]],
    constant const int& n_kernels [[buffer(6)]],
    constant const int& stride_x [[buffer(7)]],
    constant const int& stride_y [[buffer(8)]],
    constant const int& stride_z [[buffer(9)]],
    constant const int& pad_x [[buffer(10)]],
    constant const int& pad_y [[buffer(11)]],
    constant const int& pad_z [[buffer(12)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (elem == 0) {
        int out_count = 0;
        int edge_count = 0;
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
            ++out_count;
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int input_coord[4] = {
                    candidate[0],
                    candidate[1] * stride_x + offsets[offset_base] - pad_x,
                    candidate[2] * stride_y + offsets[offset_base + 1] - pad_y,
                    candidate[3] * stride_z + offsets[offset_base + 2] - pad_z,
                };
                int in_row = -1;
                if (pool_find_input_row(coords, rows, input_coord, in_row)) {
                    ++edge_count;
                }
            }
        }
        counts[0] = edge_count;
        counts[1] = out_count;
    }

    if (elem >= uint(rows)) {
        return;
    }
    int row = int(elem);
    int candidate[4];
    pool_downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
    if (pool_coord_seen(coords, row, stride_x, stride_y, stride_z, candidate)) {
        return;
    }
    int out_row = 0;
    for (int previous_row = 0; previous_row < row; ++previous_row) {
        int previous[4];
        pool_downsample_coord(
            coords, previous_row, stride_x, stride_y, stride_z, previous
        );
        if (!pool_coord_seen(
                coords, previous_row, stride_x, stride_y, stride_z, previous
            )) {
            ++out_row;
        }
    }
    write_coord(out_coords, out_row, candidate);
}

[[kernel]] void sparse_pool_identity_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant const int& n_in_rows [[buffer(5)]],
    constant const int& n_kernels [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    int rows = min(active_rows[0], n_in_rows);
    if (elem == 0) {
        int edge_count = 0;
        for (int out_row = 0; out_row < rows; ++out_row) {
            int out_base = out_row * 4;
            for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int input_coord[4] = {
                    coords[out_base],
                    coords[out_base + 1] + offsets[offset_base],
                    coords[out_base + 2] + offsets[offset_base + 1],
                    coords[out_base + 3] + offsets[offset_base + 2],
                };
                int in_row = -1;
                if (pool_find_input_row(coords, rows, input_coord, in_row)) {
                    ++edge_count;
                }
            }
        }
        counts[0] = edge_count;
        counts[1] = rows;
    }

    int coord_total = n_in_rows * 4;
    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        out_coords[elem] = row < rows ? coords[elem] : 0;
    }
}

[[kernel]] void sparse_pool_downsample_coords_hash_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device int* table_keys [[buffer(2)]],
    device int* table_rows [[buffer(3)]],
    device int* out_coords [[buffer(4)]],
    device int* counts [[buffer(5)]],
    constant const int& n_in_rows [[buffer(6)]],
    constant const int& table_capacity [[buffer(7)]],
    constant const int& empty_key [[buffer(8)]],
    constant const int& stride_x [[buffer(9)]],
    constant const int& stride_y [[buffer(10)]],
    constant const int& stride_z [[buffer(11)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    for (int slot = 0; slot < table_capacity; ++slot) {
        table_keys[slot] = empty_key;
        table_rows[slot] = -1;
    }

    int rows = min(active_rows[0], n_in_rows);
    int out_count = 0;
    for (int row = 0; row < rows; ++row) {
        int candidate[4];
        pool_downsample_coord(
            coords, row, stride_x, stride_y, stride_z, candidate
        );
        int key = pool_coord_hash(candidate);
        int slot = key & (table_capacity - 1);
        for (int probe = 0; probe < table_capacity; ++probe) {
            int found = table_keys[slot];
            if (found == empty_key) {
                table_keys[slot] = key;
                table_rows[slot] = out_count;
                write_coord(out_coords, out_count, candidate);
                ++out_count;
                break;
            }
            if (found == key) {
                int out_row = table_rows[slot];
                if (out_row >= 0 &&
                    coord4_equal(candidate, out_coords, out_row)) {
                    break;
                }
            }
            slot = (slot + 1) & (table_capacity - 1);
        }
    }
    counts[0] = rows;
    counts[1] = out_count;
}

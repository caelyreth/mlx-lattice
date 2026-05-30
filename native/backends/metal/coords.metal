#include <metal_stdlib>

using namespace metal;

[[kernel]] void fill_i32(
    device int* out [[buffer(0)]],
    constant const int& value [[buffer(1)]],
    constant const int& size [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(size)) {
        out[elem] = value;
    }
}

[[kernel]] void fill_linear_i32(
    device int* out [[buffer(0)]],
    constant const int& step [[buffer(1)]],
    constant const int& size [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(size)) {
        out[elem] = int(elem) * step;
    }
}

int coord_hash_i32(int b, int x, int y, int z) {
    uint h = 2166136261u;
    h = (h ^ uint(b)) * 16777619u;
    h = (h ^ uint(x)) * 16777619u;
    h = (h ^ uint(y)) * 16777619u;
    h = (h ^ uint(z)) * 16777619u;
    int out = int(h & 0x7fffffffu);
    return out == int(0x7fffffff) ? out - 1 : out;
}

bool same_coord(device const int* coords, int row, int b, int x, int y, int z) {
    int base = row * 4;
    return coords[base] == b && coords[base + 1] == x &&
           coords[base + 2] == y && coords[base + 3] == z;
}

[[kernel]] void insert_coord_hash_i32(
    device const int* coords [[buffer(0)]],
    device atomic_int* table_keys [[buffer(1)]],
    device int* table_rows [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& table_capacity [[buffer(4)]],
    constant const int& empty_key [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(rows)) {
        return;
    }

    int base = int(row) * 4;
    int b = coords[base];
    int x = coords[base + 1];
    int y = coords[base + 2];
    int z = coords[base + 3];
    int key = coord_hash_i32(b, x, y, z);
    int slot = key & (table_capacity - 1);

    for (int probe = 0; probe < table_capacity; ++probe) {
        int expected = empty_key;
        if (atomic_compare_exchange_weak_explicit(
                &table_keys[slot],
                &expected,
                key,
                memory_order_relaxed,
                memory_order_relaxed
            )) {
            table_rows[slot] = int(row);
            return;
        }
        if (expected == key) {
            int existing = table_rows[slot];
            if (existing >= 0 && same_coord(coords, existing, b, x, y, z)) {
                return;
            }
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

[[kernel]] void build_subm_kernel_map_i32(
    device const int* coords [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device const int* table_keys [[buffer(2)]],
    device const int* table_rows [[buffer(3)]],
    device int* maps [[buffer(4)]],
    device atomic_int* sizes [[buffer(5)]],
    device int* kernels [[buffer(6)]],
    device int* residual_maps [[buffer(7)]],
    device int* residual_kernels [[buffer(8)]],
    constant const int& rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& center_kernel [[buffer(11)]],
    constant const int& table_capacity [[buffer(12)]],
    constant const int& empty_key [[buffer(13)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(rows * kernel_count);
    if (elem >= total) {
        return;
    }

    int kernel_index = int(elem / uint(rows));
    int out_row = int(elem - uint(kernel_index * rows));
    int base = out_row * 4;
    int target_b = coords[base];
    int target_x = coords[base + 1] + offsets[kernel_index * 3];
    int target_y = coords[base + 2] + offsets[kernel_index * 3 + 1];
    int target_z = coords[base + 3] + offsets[kernel_index * 3 + 2];
    int key = coord_hash_i32(target_b, target_x, target_y, target_z);
    int slot = key & (table_capacity - 1);

    for (int probe = 0; probe < table_capacity; ++probe) {
        int found_key = table_keys[slot];
        if (found_key == empty_key) {
            return;
        }
        if (found_key == key) {
            int in_row = table_rows[slot];
            if (in_row < 0 ||
                !same_coord(
                    coords, in_row, target_b, target_x, target_y, target_z
                )) {
                slot = (slot + 1) & (table_capacity - 1);
                continue;
            }
            maps[elem * 2] = in_row;
            maps[elem * 2 + 1] = out_row;
            kernels[elem] = kernel_index;
            atomic_fetch_add_explicit(
                &sizes[kernel_index], 1, memory_order_relaxed
            );
            if (kernel_index != center_kernel) {
                int slot = kernel_index < center_kernel ? kernel_index
                                                        : kernel_index - 1;
                int residual = out_row * (kernel_count - 1) + slot;
                residual_maps[residual * 2] = in_row;
                residual_maps[residual * 2 + 1] = out_row;
                residual_kernels[residual] = kernel_index;
            }
            return;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

[[kernel]] void build_generative_map_i32(
    device const int* coords [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device int* maps [[buffer(2)]],
    device int* kernels [[buffer(3)]],
    device int* out_coords [[buffer(4)]],
    constant const int& rows [[buffer(5)]],
    constant const int& kernel_count [[buffer(6)]],
    constant const int& stride_x [[buffer(7)]],
    constant const int& stride_y [[buffer(8)]],
    constant const int& stride_z [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(rows * kernel_count);
    if (elem >= total) {
        return;
    }

    int in_row = int(elem / uint(kernel_count));
    int kernel_index = int(elem - uint(in_row * kernel_count));
    int out_row = int(elem);
    int in_base = in_row * 4;
    int out_base = out_row * 4;
    maps[out_row * 2] = in_row;
    maps[out_row * 2 + 1] = out_row;
    kernels[out_row] = kernel_index;
    out_coords[out_base] = coords[in_base];
    out_coords[out_base + 1] =
        coords[in_base + 1] * stride_x + offsets[kernel_index * 3];
    out_coords[out_base + 2] =
        coords[in_base + 2] * stride_y + offsets[kernel_index * 3 + 1];
    out_coords[out_base + 3] =
        coords[in_base + 3] * stride_z + offsets[kernel_index * 3 + 2];
}

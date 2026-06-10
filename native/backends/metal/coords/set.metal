#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/coords/common.metal"

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

inline int lookup_downsample_row_hash(
    device const int* coords,
    device const int* table_rows,
    int table_capacity,
    thread const int* target,
    int stride_x,
    int stride_y,
    int stride_z
) {
    int slot = coord_hash_i32(target) & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int row = table_rows[slot];
        if (row < 0) {
            return -1;
        }
        int stored[4];
        downsample_coord(coords, row, stride_x, stride_y, stride_z, stored);
        if (target[0] == stored[0] && target[1] == stored[1] &&
            target[2] == stored[2] && target[3] == stored[3]) {
            return row;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
    return -1;
}

[[kernel]] void build_downsample_coord_hash_i32(
    device const int* coords [[buffer(0)]],
    device atomic_int* table_rows [[buffer(1)]],
    constant const int& rows [[buffer(2)]],
    constant const int& table_capacity [[buffer(3)]],
    constant const int& stride_x [[buffer(4)]],
    constant const int& stride_y [[buffer(5)]],
    constant const int& stride_z [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(rows)) {
        return;
    }

    int candidate[4];
    downsample_coord(coords, int(row), stride_x, stride_y, stride_z, candidate);
    int slot = coord_hash_i32(candidate) & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(
                &table_rows[slot],
                &expected,
                int(row),
                memory_order_relaxed,
                memory_order_relaxed
            )) {
            return;
        }
        int stored[4];
        downsample_coord(
            coords, expected, stride_x, stride_y, stride_z, stored
        );
        if (candidate[0] == stored[0] && candidate[1] == stored[1] &&
            candidate[2] == stored[2] && candidate[3] == stored[3]) {
            atomic_fetch_min_explicit(
                &table_rows[slot], int(row), memory_order_relaxed
            );
            return;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

[[kernel]] void plan_downsample_coord_set_i32(
    device const int* coords [[buffer(0)]],
    device const int* table_rows [[buffer(1)]],
    device int* selected [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& table_capacity [[buffer(4)]],
    constant const int& stride_x [[buffer(5)]],
    constant const int& stride_y [[buffer(6)]],
    constant const int& stride_z [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(rows)) {
        return;
    }
    int candidate[4];
    downsample_coord(coords, int(row), stride_x, stride_y, stride_z, candidate);
    selected[row] = lookup_downsample_row_hash(
                        coords,
                        table_rows,
                        table_capacity,
                        candidate,
                        stride_x,
                        stride_y,
                        stride_z
                    ) == int(row);
}

[[kernel]] void compact_downsample_coord_set_i32(
    device const int* coords [[buffer(0)]],
    device const int* selected [[buffer(1)]],
    device int* out_coords [[buffer(2)]],
    device int* count [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& stride_x [[buffer(5)]],
    constant const int& stride_y [[buffer(6)]],
    constant const int& stride_z [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int out_count = 0;
    for (int row = 0; row < rows; ++row) {
        if (selected[row] == 0) {
            continue;
        }
        int candidate[4];
        downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
        write_coord(out_coords, out_count++, candidate);
    }
    count[0] = out_count;
}

[[kernel]] void compact_strided_relation_output_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* selected [[buffer(1)]],
    device int* out_coords [[buffer(2)]],
    device int* counts [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& stride_x [[buffer(5)]],
    constant const int& stride_y [[buffer(6)]],
    constant const int& stride_z [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int out_count = 0;
    for (int row = 0; row < rows; ++row) {
        if (selected[row] == 0) {
            continue;
        }
        int candidate[4];
        downsample_coord(coords, row, stride_x, stride_y, stride_z, candidate);
        write_coord(out_coords, out_count++, candidate);
    }
    counts[0] = 0;
    counts[1] = out_count;
}

[[kernel]] void build_strided_relation_output_hash_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device atomic_int* table_rows [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& table_capacity [[buffer(4)]],
    constant const int& stride_x [[buffer(5)]],
    constant const int& stride_y [[buffer(6)]],
    constant const int& stride_z [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    if (row >= uint(logical_rows)) {
        return;
    }

    int candidate[4];
    downsample_coord(coords, int(row), stride_x, stride_y, stride_z, candidate);
    int slot = coord_hash_i32(candidate) & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(
                &table_rows[slot],
                &expected,
                int(row),
                memory_order_relaxed,
                memory_order_relaxed
            )) {
            return;
        }
        int stored[4];
        downsample_coord(
            coords, expected, stride_x, stride_y, stride_z, stored
        );
        if (candidate[0] == stored[0] && candidate[1] == stored[1] &&
            candidate[2] == stored[2] && candidate[3] == stored[3]) {
            atomic_fetch_min_explicit(
                &table_rows[slot], int(row), memory_order_relaxed
            );
            return;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
}

[[kernel]] void plan_strided_relation_output_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* table_rows [[buffer(2)]],
    device int* selected [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& table_capacity [[buffer(5)]],
    constant const int& stride_x [[buffer(6)]],
    constant const int& stride_y [[buffer(7)]],
    constant const int& stride_z [[buffer(8)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(rows)) {
        return;
    }
    int logical_rows = min(active_rows[0], rows);
    if (row >= uint(logical_rows)) {
        selected[row] = 0;
        return;
    }

    int candidate[4];
    downsample_coord(coords, int(row), stride_x, stride_y, stride_z, candidate);
    selected[row] = lookup_downsample_row_hash(
                        coords,
                        table_rows,
                        table_capacity,
                        candidate,
                        stride_x,
                        stride_y,
                        stride_z
                    ) == int(row);
}

[[kernel]] void plan_union_coord_set_i32(
    device const int* lhs [[buffer(0)]],
    device const int* rhs [[buffer(1)]],
    device const int* lhs_table_rows [[buffer(2)]],
    device const int* rhs_table_rows [[buffer(3)]],
    device int* selected [[buffer(4)]],
    constant const int& lhs_rows [[buffer(5)]],
    constant const int& rhs_rows [[buffer(6)]],
    constant const int& lhs_table_capacity [[buffer(7)]],
    constant const int& rhs_table_capacity [[buffer(8)]],
    uint row [[thread_position_in_grid]]
) {
    int total_rows = lhs_rows + rhs_rows;
    if (row >= uint(total_rows)) {
        return;
    }
    device const int* source = row < uint(lhs_rows) ? lhs : rhs;
    int source_row = row < uint(lhs_rows) ? int(row) : int(row) - lhs_rows;
    int base = source_row * 4;
    int candidate[4] = {
        source[base],
        source[base + 1],
        source[base + 2],
        source[base + 3],
    };
    if (row < uint(lhs_rows)) {
        selected[row] = lookup_coord_row_hash(
                            lhs, lhs_table_rows, lhs_table_capacity, candidate
                        ) == source_row;
        return;
    }
    selected[row] = lookup_coord_row_hash(
                        lhs, lhs_table_rows, lhs_table_capacity, candidate
                    ) < 0 &&
                    lookup_coord_row_hash(
                        rhs, rhs_table_rows, rhs_table_capacity, candidate
                    ) == source_row;
}

[[kernel]] void compact_union_coord_set_i32(
    device const int* lhs [[buffer(0)]],
    device const int* rhs [[buffer(1)]],
    device const int* selected [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* count [[buffer(4)]],
    constant const int& lhs_rows [[buffer(5)]],
    constant const int& rhs_rows [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int out_count = 0;
    int total_rows = lhs_rows + rhs_rows;
    for (int row = 0; row < total_rows; ++row) {
        if (selected[row] == 0) {
            continue;
        }
        device const int* source = row < lhs_rows ? lhs : rhs;
        int source_row = row < lhs_rows ? row : row - lhs_rows;
        int base = source_row * 4;
        int candidate[4] = {
            source[base],
            source[base + 1],
            source[base + 2],
            source[base + 3],
        };
        write_coord(out_coords, out_count++, candidate);
    }
    count[0] = out_count;
}

[[kernel]] void plan_intersection_coord_set_i32(
    device const int* lhs [[buffer(0)]],
    device const int* rhs [[buffer(1)]],
    device const int* rhs_table_rows [[buffer(2)]],
    device const int* lhs_table_rows [[buffer(3)]],
    device int* selected [[buffer(4)]],
    constant const int& lhs_rows [[buffer(5)]],
    constant const int& rhs_table_capacity [[buffer(6)]],
    constant const int& lhs_table_capacity [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= uint(lhs_rows)) {
        return;
    }
    int base = int(row) * 4;
    int candidate[4] = {
        lhs[base],
        lhs[base + 1],
        lhs[base + 2],
        lhs[base + 3],
    };
    selected[row] = lookup_coord_row_hash(
                        lhs, lhs_table_rows, lhs_table_capacity, candidate
                    ) == int(row) &&
                    lookup_coord_row_hash(
                        rhs, rhs_table_rows, rhs_table_capacity, candidate
                    ) >= 0;
}

[[kernel]] void compact_intersection_coord_set_i32(
    device const int* lhs [[buffer(0)]],
    device const int* selected [[buffer(1)]],
    device int* out_coords [[buffer(2)]],
    device int* count [[buffer(3)]],
    constant const int& lhs_rows [[buffer(4)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int out_count = 0;
    for (int row = 0; row < lhs_rows; ++row) {
        if (selected[row] == 0) {
            continue;
        }
        int base = row * 4;
        int candidate[4] = {
            lhs[base],
            lhs[base + 1],
            lhs[base + 2],
            lhs[base + 3],
        };
        write_coord(out_coords, out_count++, candidate);
    }
    count[0] = out_count;
}

[[kernel]] void lookup_coords_hash_i32(
    device const int* coords [[buffer(0)]],
    device const int* queries [[buffer(1)]],
    device const int* table_rows [[buffer(2)]],
    device int* out_rows [[buffer(3)]],
    constant const int& query_rows [[buffer(4)]],
    constant const int& table_capacity [[buffer(5)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem >= uint(query_rows)) {
        return;
    }

    int query_row = int(elem);
    int base = query_row * 4;
    int candidate[4] = {
        queries[base],
        queries[base + 1],
        queries[base + 2],
        queries[base + 3],
    };
    out_rows[query_row] =
        lookup_coord_row_hash(coords, table_rows, table_capacity, candidate);
}

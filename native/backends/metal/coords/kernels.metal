#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/coords/common.metal"

// MARK: - generative relations

[[kernel]] void build_generative_kernel_relation_i32(
    device const int* coords [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* in_rows [[buffer(3)]],
    device int* out_rows [[buffer(4)]],
    device int* kernel_ids [[buffer(5)]],
    device int* row_offsets [[buffer(6)]],
    device int* out_coords [[buffer(7)]],
    device int* counts [[buffer(8)]],
    constant const int& rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& stride_x [[buffer(11)]],
    constant const int& stride_y [[buffer(12)]],
    constant const int& stride_z [[buffer(13)]],
    uint elem [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    uint total = uint(logical_rows * kernel_count);
    if (elem == 0) {
        counts[0] = int(total);
        counts[1] = int(total);
        row_offsets[total] = int(total);
    }
    if (elem >= total) {
        return;
    }

    int out_row = int(elem);
    int in_row = int(elem / uint(kernel_count));
    int kernel_index = int(elem - uint(in_row * kernel_count));
    int in_base = in_row * 4;
    int out_base = out_row * 4;
    int offset_base = kernel_index * 3;

    in_rows[out_row] = in_row;
    out_rows[out_row] = out_row;
    kernel_ids[out_row] = kernel_index;
    row_offsets[out_row] = out_row;
    out_coords[out_base] = coords[in_base];
    out_coords[out_base + 1] =
        coords[in_base + 1] * stride_x + offsets[offset_base];
    out_coords[out_base + 2] =
        coords[in_base + 2] * stride_y + offsets[offset_base + 1];
    out_coords[out_base + 3] =
        coords[in_base + 3] * stride_z + offsets[offset_base + 2];
}

// MARK: - generic relations

[[kernel]] void build_identity_forward_relation_plan_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const int* table_rows [[buffer(3)]],
    device int* planned_in_rows [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& kernel_count [[buffer(7)]],
    constant const int& table_capacity [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    int coord_total = rows * 4;
    if (elem < uint(coord_total)) {
        int row = int(elem) / 4;
        out_coords[elem] = row < logical_rows ? coords[elem] : 0;
    }

    int relation_total = rows * kernel_count;
    if (elem >= uint(relation_total)) {
        return;
    }
    int kernel_id = int(elem) / rows;
    int out_row = int(elem) - kernel_id * rows;
    if (out_row >= logical_rows) {
        planned_in_rows[elem] = -1;
        return;
    }

    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int candidate[4] = {
        coords[out_base],
        coords[out_base + 1] + kernel_offsets[offset_base],
        coords[out_base + 2] + kernel_offsets[offset_base + 1],
        coords[out_base + 3] + kernel_offsets[offset_base + 2],
    };
    planned_in_rows[elem] =
        lookup_coord_row_hash(coords, table_rows, table_capacity, candidate);
}

[[kernel]] void build_identity_forward_relation_compact_i32(
    device const int* planned_in_rows [[buffer(0)]],
    device int* in_rows [[buffer(1)]],
    device int* out_rows [[buffer(2)]],
    device int* kernel_ids [[buffer(3)]],
    device int* row_offsets [[buffer(4)]],
    device int* counts [[buffer(5)]],
    device const int* active_rows [[buffer(6)]],
    constant const int& rows [[buffer(7)]],
    constant const int& kernel_count [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int edge_count = 0;
    int out_count = min(active_rows[0], rows);
    for (int out_row = 0; out_row < out_count; ++out_row) {
        row_offsets[out_row] = edge_count;
        for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
            int kernel_base = kernel_id * rows;
            int in_row = planned_in_rows[kernel_base + out_row];
            if (in_row < 0) {
                continue;
            }
            write_edge(
                in_rows,
                out_rows,
                kernel_ids,
                edge_count,
                in_row,
                out_row,
                kernel_id
            );
            edge_count += 1;
        }
    }
    for (int out_row = out_count; out_row <= rows; ++out_row) {
        row_offsets[out_row] = edge_count;
    }
    counts[0] = edge_count;
    counts[1] = out_count;
}

[[kernel]] void build_strided_forward_output_coords_i32(
    device const int* coords [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device int* table_rows [[buffer(2)]],
    device int* out_coords [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant const int& rows [[buffer(5)]],
    constant const int& table_capacity [[buffer(6)]],
    constant const int& stride_x [[buffer(7)]],
    constant const int& stride_y [[buffer(8)]],
    constant const int& stride_z [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int logical_rows = min(active_rows[0], rows);
    int out_count = 0;
    for (int row = 0; row < logical_rows; ++row) {
        int base = row * 4;
        int candidate[4] = {
            coords[base],
            floor_div_int(coords[base + 1], stride_x),
            floor_div_int(coords[base + 2], stride_y),
            floor_div_int(coords[base + 3], stride_z),
        };
        int slot = coord_hash_i32(candidate) & (table_capacity - 1);
        for (int probe = 0; probe < table_capacity; ++probe) {
            int out_row = table_rows[slot];
            if (out_row < 0) {
                table_rows[slot] = out_count;
                write_coord(out_coords, out_count, candidate);
                out_count += 1;
                break;
            }
            if (coord4_equal(candidate, out_coords, out_row)) {
                break;
            }
            slot = (slot + 1) & (table_capacity - 1);
        }
    }
    counts[1] = out_count;
}

[[kernel]] void build_strided_forward_relation_plan_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* out_coords [[buffer(2)]],
    device const int* counts [[buffer(3)]],
    device const int* table_rows [[buffer(4)]],
    device int* planned_in_rows [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& kernel_count [[buffer(7)]],
    constant const int& table_capacity [[buffer(8)]],
    constant const int& stride_x [[buffer(9)]],
    constant const int& stride_y [[buffer(10)]],
    constant const int& stride_z [[buffer(11)]],
    constant const int& pad_x [[buffer(12)]],
    constant const int& pad_y [[buffer(13)]],
    constant const int& pad_z [[buffer(14)]],
    uint elem [[thread_position_in_grid]]
) {
    int relation_total = rows * kernel_count;
    if (elem >= uint(relation_total)) {
        return;
    }

    int kernel_id = int(elem) / rows;
    int out_row = int(elem) - kernel_id * rows;
    int out_count = min(counts[1], rows);
    if (out_row >= out_count) {
        planned_in_rows[elem] = -1;
        return;
    }

    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;
    int candidate[4] = {
        out_coords[out_base],
        out_coords[out_base + 1] * stride_x + kernel_offsets[offset_base] -
            pad_x,
        out_coords[out_base + 2] * stride_y + kernel_offsets[offset_base + 1] -
            pad_y,
        out_coords[out_base + 3] * stride_z + kernel_offsets[offset_base + 2] -
            pad_z,
    };
    planned_in_rows[elem] =
        lookup_coord_row_hash(coords, table_rows, table_capacity, candidate);
}

[[kernel]] void build_strided_forward_relation_compact_i32(
    device const int* planned_in_rows [[buffer(0)]],
    device int* in_rows [[buffer(1)]],
    device int* out_rows [[buffer(2)]],
    device int* kernel_ids [[buffer(3)]],
    device int* row_offsets [[buffer(4)]],
    device int* counts [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& kernel_count [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int edge_count = 0;
    int out_count = min(counts[1], rows);
    for (int out_row = 0; out_row < out_count; ++out_row) {
        row_offsets[out_row] = edge_count;
        for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
            int kernel_base = kernel_id * rows;
            int in_row = planned_in_rows[kernel_base + out_row];
            if (in_row < 0) {
                continue;
            }
            write_edge(
                in_rows,
                out_rows,
                kernel_ids,
                edge_count,
                in_row,
                out_row,
                kernel_id
            );
            edge_count += 1;
        }
    }
    for (int out_row = out_count; out_row <= rows; ++out_row) {
        row_offsets[out_row] = edge_count;
    }
    counts[0] = edge_count;
}

[[kernel]] void build_transposed_direct_relation_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* in_rows [[buffer(3)]],
    device int* out_rows [[buffer(4)]],
    device int* kernel_ids [[buffer(5)]],
    device int* row_offsets [[buffer(6)]],
    device int* out_coords [[buffer(7)]],
    device int* counts [[buffer(8)]],
    constant const int& rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& stride_x [[buffer(11)]],
    constant const int& stride_y [[buffer(12)]],
    constant const int& stride_z [[buffer(13)]],
    constant const int& pad_x [[buffer(14)]],
    constant const int& pad_y [[buffer(15)]],
    constant const int& pad_z [[buffer(16)]],
    uint elem [[thread_position_in_grid]]
) {
    int logical_rows = min(active_rows[0], rows);
    int total = logical_rows * kernel_count;
    if (elem == 0) {
        counts[0] = total;
        counts[1] = total;
        row_offsets[total] = total;
    }
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem);
    int in_row = int(elem) / kernel_count;
    int kernel_id = int(elem) - in_row * kernel_count;
    int in_base = in_row * 4;
    int out_base = out_row * 4;
    int offset_base = kernel_id * 3;

    in_rows[out_row] = in_row;
    out_rows[out_row] = out_row;
    kernel_ids[out_row] = kernel_id;
    row_offsets[out_row] = out_row;
    out_coords[out_base] = coords[in_base];
    out_coords[out_base + 1] =
        coords[in_base + 1] * stride_x + kernel_offsets[offset_base] - pad_x;
    out_coords[out_base + 2] = coords[in_base + 2] * stride_y +
                               kernel_offsets[offset_base + 1] - pad_y;
    out_coords[out_base + 3] = coords[in_base + 3] * stride_z +
                               kernel_offsets[offset_base + 2] - pad_z;
}

[[kernel]] void build_transposed_kernel_relation_i32(
    device const int* coords [[buffer(0)]],
    device const int* kernel_offsets [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device int* in_rows [[buffer(3)]],
    device int* out_rows [[buffer(4)]],
    device int* kernel_ids [[buffer(5)]],
    device int* row_offsets [[buffer(6)]],
    device int* out_coords [[buffer(7)]],
    device int* counts [[buffer(8)]],
    constant const int& rows [[buffer(9)]],
    constant const int& kernel_count [[buffer(10)]],
    constant const int& stride_x [[buffer(11)]],
    constant const int& stride_y [[buffer(12)]],
    constant const int& stride_z [[buffer(13)]],
    constant const int& pad_x [[buffer(14)]],
    constant const int& pad_y [[buffer(15)]],
    constant const int& pad_z [[buffer(16)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_count = 0;
    int logical_rows = min(active_rows[0], rows);
    for (int in_row = 0; in_row < logical_rows; ++in_row) {
        int in_base = in_row * 4;
        for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
            int offset_base = kernel_id * 3;
            int candidate[4] = {
                coords[in_base],
                coords[in_base + 1] * stride_x + kernel_offsets[offset_base] -
                    pad_x,
                coords[in_base + 2] * stride_y +
                    kernel_offsets[offset_base + 1] - pad_y,
                coords[in_base + 3] * stride_z +
                    kernel_offsets[offset_base + 2] - pad_z,
            };

            int out_row = -1;
            for (int prev = 0; prev < out_count; ++prev) {
                if (coord4_equal(candidate, out_coords, prev)) {
                    out_row = prev;
                    break;
                }
            }
            if (out_row < 0) {
                out_row = out_count;
                write_coord(out_coords, out_row, candidate);
                out_count += 1;
            }
        }
    }

    int edge_count = 0;
    for (int out_row = 0; out_row < out_count; ++out_row) {
        row_offsets[out_row] = edge_count;
        for (int in_row = 0; in_row < logical_rows; ++in_row) {
            int in_base = in_row * 4;
            for (int kernel_id = 0; kernel_id < kernel_count; ++kernel_id) {
                int offset_base = kernel_id * 3;
                int candidate[4] = {
                    coords[in_base],
                    coords[in_base + 1] * stride_x +
                        kernel_offsets[offset_base] - pad_x,
                    coords[in_base + 2] * stride_y +
                        kernel_offsets[offset_base + 1] - pad_y,
                    coords[in_base + 3] * stride_z +
                        kernel_offsets[offset_base + 2] - pad_z,
                };
                if (!coord4_equal(candidate, out_coords, out_row)) {
                    continue;
                }
                write_edge(
                    in_rows,
                    out_rows,
                    kernel_ids,
                    edge_count,
                    in_row,
                    out_row,
                    kernel_id
                );
                edge_count += 1;
            }
        }
    }
    for (int out_row = out_count; out_row <= rows * kernel_count; ++out_row) {
        row_offsets[out_row] = edge_count;
    }
    counts[0] = edge_count;
    counts[1] = out_count;
}

// MARK: - neighbor relations

[[kernel]] void build_neighbor_relation_i32(
    device const int* query_active_rows [[buffer(0)]],
    device int* query_rows [[buffer(1)]],
    device int* source_rows [[buffer(2)]],
    device int* neighbor_ids [[buffer(3)]],
    device float* distances [[buffer(4)]],
    device int* row_offsets [[buffer(5)]],
    device int* counts [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_capacity = query_capacity * max_neighbors;
    int query_count = min(query_active_rows[0], query_capacity);

    if (elem == 0) {
        counts[0] = 0;
        counts[1] = query_count;
    }

    if (elem <= uint(query_capacity)) {
        row_offsets[elem] = 0;
    }

    if (elem >= uint(edge_capacity)) {
        return;
    }

    query_rows[elem] = 0;
    source_rows[elem] = -1;
    neighbor_ids[elem] = 0;
    distances[elem] = 0.0f;
}

[[kernel]] void fill_neighbor_relation_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device int* query_rows [[buffer(4)]],
    device int* source_rows [[buffer(5)]],
    device int* neighbor_ids [[buffer(6)]],
    device float* distances [[buffer(7)]],
    constant const int& op [[buffer(8)]],
    constant const int& source_capacity [[buffer(9)]],
    constant const int& query_capacity [[buffer(10)]],
    constant const int& max_neighbors [[buffer(11)]],
    constant const float& radius_squared [[buffer(12)]],
    uint query_row [[thread_position_in_grid]]
) {
    int source_count = min(source_active_rows[0], source_capacity);
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int slot_start = int(query_row) * max_neighbors;
    int selected = 0;
    for (int source_row = 0; source_row < source_count; ++source_row) {
        if (!same_batch(
                query_coords, int(query_row), source_coords, source_row
            )) {
            continue;
        }
        float distance = squared_spatial_distance(
            query_coords, int(query_row), source_coords, source_row
        );
        if (op == 1 && distance > radius_squared) {
            continue;
        }

        int insert_at = selected;
        for (int rank = 0; rank < selected; ++rank) {
            int index = slot_start + rank;
            float existing_distance = distances[index];
            int existing_source = source_rows[index];
            if (distance < existing_distance ||
                (distance == existing_distance &&
                 source_row < existing_source)) {
                insert_at = rank;
                break;
            }
        }
        if (insert_at >= max_neighbors) {
            continue;
        }

        int last = min(selected, max_neighbors - 1);
        for (int rank = last; rank > insert_at; --rank) {
            int dst = slot_start + rank;
            int src = dst - 1;
            source_rows[dst] = source_rows[src];
            distances[dst] = distances[src];
        }
        source_rows[slot_start + insert_at] = source_row;
        distances[slot_start + insert_at] = distance;
        selected = min(selected + 1, max_neighbors);
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void fill_radius_relation_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device int* query_rows [[buffer(5)]],
    device int* source_rows [[buffer(6)]],
    device int* neighbor_ids [[buffer(7)]],
    device float* distances [[buffer(8)]],
    constant const int& source_capacity [[buffer(9)]],
    constant const int& query_capacity [[buffer(10)]],
    constant const int& max_neighbors [[buffer(11)]],
    constant const float& radius_squared [[buffer(12)]],
    constant const int& ceil_radius [[buffer(13)]],
    constant const int& table_capacity [[buffer(14)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int query_base = int(query_row) * 4;
    int slot_start = int(query_row) * max_neighbors;
    int selected = 0;
    for (int dz = -ceil_radius; dz <= ceil_radius; ++dz) {
        for (int dy = -ceil_radius; dy <= ceil_radius; ++dy) {
            for (int dx = -ceil_radius; dx <= ceil_radius; ++dx) {
                float distance = float(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }

                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }

                int insert_at = selected;
                for (int rank = 0; rank < selected; ++rank) {
                    int index = slot_start + rank;
                    float existing_distance = distances[index];
                    int existing_source = source_rows[index];
                    if (distance < existing_distance ||
                        (distance == existing_distance &&
                         source_row < existing_source)) {
                        insert_at = rank;
                        break;
                    }
                }
                if (insert_at >= max_neighbors) {
                    continue;
                }

                int last = min(selected, max_neighbors - 1);
                for (int rank = last; rank > insert_at; --rank) {
                    int dst = slot_start + rank;
                    int src = dst - 1;
                    source_rows[dst] = source_rows[src];
                    distances[dst] = distances[src];
                }
                source_rows[slot_start + insert_at] = source_row;
                distances[slot_start + insert_at] = distance;
                selected = min(selected + 1, max_neighbors);
            }
        }
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void count_radius_relation_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device int* row_counts [[buffer(5)]],
    constant const int& source_capacity [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    constant const float& radius_squared [[buffer(9)]],
    constant const int& ceil_radius [[buffer(10)]],
    constant const int& table_capacity [[buffer(11)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_capacity)) {
        return;
    }
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        row_counts[query_row] = 0;
        return;
    }

    int query_base = int(query_row) * 4;
    int selected = 0;
    for (int dz = -ceil_radius; dz <= ceil_radius; ++dz) {
        for (int dy = -ceil_radius; dy <= ceil_radius; ++dy) {
            for (int dx = -ceil_radius; dx <= ceil_radius; ++dx) {
                float distance = float(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }

                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }
                selected += 1;
                if (selected >= max_neighbors) {
                    row_counts[query_row] = max_neighbors;
                    return;
                }
            }
        }
    }
    row_counts[query_row] = selected;
}

[[kernel]] void prefix_neighbor_row_offsets_i32(
    device int* row_offsets [[buffer(0)]],
    device int* counts [[buffer(1)]],
    constant const int& query_capacity [[buffer(2)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int query_count = counts[1];
    int total = 0;
    for (int query_row = 0; query_row < query_count; ++query_row) {
        int row_count = row_offsets[query_row];
        row_offsets[query_row] = total;
        total += row_count;
    }
    row_offsets[query_count] = total;
    counts[0] = total;
    (void)query_capacity;
}

[[kernel]] void fill_radius_relation_compact_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device const int* row_offsets [[buffer(5)]],
    device int* query_rows [[buffer(6)]],
    device int* source_rows [[buffer(7)]],
    device int* neighbor_ids [[buffer(8)]],
    device float* distances [[buffer(9)]],
    constant const int& source_capacity [[buffer(10)]],
    constant const int& query_capacity [[buffer(11)]],
    constant const int& max_neighbors [[buffer(12)]],
    constant const float& radius_squared [[buffer(13)]],
    constant const int& ceil_radius [[buffer(14)]],
    constant const int& table_capacity [[buffer(15)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int query_base = int(query_row) * 4;
    int slot_start = row_offsets[query_row];
    int selected = 0;
    for (int dz = -ceil_radius; dz <= ceil_radius; ++dz) {
        for (int dy = -ceil_radius; dy <= ceil_radius; ++dy) {
            for (int dx = -ceil_radius; dx <= ceil_radius; ++dx) {
                float distance = float(dx * dx + dy * dy + dz * dz);
                if (distance > radius_squared) {
                    continue;
                }

                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }

                int insert_at = selected;
                for (int rank = 0; rank < selected; ++rank) {
                    int index = slot_start + rank;
                    float existing_distance = distances[index];
                    int existing_source = source_rows[index];
                    if (distance < existing_distance ||
                        (distance == existing_distance &&
                         source_row < existing_source)) {
                        insert_at = rank;
                        break;
                    }
                }
                if (insert_at >= max_neighbors) {
                    continue;
                }

                int last = min(selected, max_neighbors - 1);
                for (int rank = last; rank > insert_at; --rank) {
                    int dst = slot_start + rank;
                    int src = dst - 1;
                    source_rows[dst] = source_rows[src];
                    distances[dst] = distances[src];
                }
                source_rows[slot_start + insert_at] = source_row;
                distances[slot_start + insert_at] = distance;
                selected = min(selected + 1, max_neighbors);
            }
        }
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void count_knn_relation_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device int* row_counts [[buffer(5)]],
    constant const int& source_capacity [[buffer(6)]],
    constant const int& query_capacity [[buffer(7)]],
    constant const int& max_neighbors [[buffer(8)]],
    constant const int& search_radius [[buffer(9)]],
    constant const int& table_capacity [[buffer(10)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_capacity)) {
        return;
    }
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        row_counts[query_row] = 0;
        return;
    }

    int query_base = int(query_row) * 4;
    int selected = 0;
    for (int dz = -search_radius; dz <= search_radius; ++dz) {
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dx = -search_radius; dx <= search_radius; ++dx) {
                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }
                selected += 1;
                if (selected >= max_neighbors) {
                    row_counts[query_row] = max_neighbors;
                    return;
                }
            }
        }
    }
    row_counts[query_row] = selected;
}

[[kernel]] void fill_knn_relation_compact_hash_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device const int* source_table [[buffer(4)]],
    device const int* row_offsets [[buffer(5)]],
    device int* query_rows [[buffer(6)]],
    device int* source_rows [[buffer(7)]],
    device int* neighbor_ids [[buffer(8)]],
    device float* distances [[buffer(9)]],
    constant const int& source_capacity [[buffer(10)]],
    constant const int& query_capacity [[buffer(11)]],
    constant const int& max_neighbors [[buffer(12)]],
    constant const int& search_radius [[buffer(13)]],
    constant const int& table_capacity [[buffer(14)]],
    uint query_row [[thread_position_in_grid]]
) {
    int query_count = min(query_active_rows[0], query_capacity);
    if (query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int query_base = int(query_row) * 4;
    int slot_start = row_offsets[query_row];
    int selected = 0;
    for (int dz = -search_radius; dz <= search_radius; ++dz) {
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dx = -search_radius; dx <= search_radius; ++dx) {
                int target[4];
                target[0] = query_coords[query_base];
                target[1] = query_coords[query_base + 1] + dx;
                target[2] = query_coords[query_base + 2] + dy;
                target[3] = query_coords[query_base + 3] + dz;
                int source_row = lookup_coord_row_hash(
                    source_coords, source_table, table_capacity, target
                );
                if (source_row < 0 || source_row >= source_capacity ||
                    source_row >= source_active_rows[0]) {
                    continue;
                }

                float distance = float(dx * dx + dy * dy + dz * dz);
                int insert_at = selected;
                for (int rank = 0; rank < selected; ++rank) {
                    int index = slot_start + rank;
                    float existing_distance = distances[index];
                    int existing_source = source_rows[index];
                    if (distance < existing_distance ||
                        (distance == existing_distance &&
                         source_row < existing_source)) {
                        insert_at = rank;
                        break;
                    }
                }
                if (insert_at >= max_neighbors) {
                    continue;
                }

                int last = min(selected, max_neighbors - 1);
                for (int rank = last; rank > insert_at; --rank) {
                    int dst = slot_start + rank;
                    int src = dst - 1;
                    source_rows[dst] = source_rows[src];
                    distances[dst] = distances[src];
                }
                source_rows[slot_start + insert_at] = source_row;
                distances[slot_start + insert_at] = distance;
                selected = min(selected + 1, max_neighbors);
            }
        }
    }

    for (int rank = 0; rank < selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void fill_knn_relation_topk_i32(
    device const int* source_coords [[buffer(0)]],
    device const int* query_coords [[buffer(1)]],
    device const int* source_active_rows [[buffer(2)]],
    device const int* query_active_rows [[buffer(3)]],
    device int* query_rows [[buffer(4)]],
    device int* source_rows [[buffer(5)]],
    device int* neighbor_ids [[buffer(6)]],
    device float* distances [[buffer(7)]],
    constant const int& source_capacity [[buffer(8)]],
    constant const int& query_capacity [[buffer(9)]],
    constant const int& max_neighbors [[buffer(10)]],
    uint query_row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]]
) {
    constexpr int thread_count = 128;
    constexpr int max_k = 16;
    threadgroup float group_distances[thread_count * max_k];
    threadgroup int group_sources[thread_count * max_k];

    int tid = int(thread_id);
    int source_count = min(source_active_rows[0], source_capacity);
    int query_count = min(query_active_rows[0], query_capacity);
    int k = min(max_neighbors, max_k);
    int slot_start = int(query_row) * max_neighbors;

    float local_distances[max_k];
    int local_sources[max_k];
    for (int rank = 0; rank < max_k; ++rank) {
        local_distances[rank] = 0.0f;
        local_sources[rank] = -1;
    }

    int selected = 0;
    if (query_row < uint(query_count) && max_neighbors > 0) {
        for (int source_row = tid; source_row < source_count;
             source_row += thread_count) {
            if (!same_batch(
                    query_coords, int(query_row), source_coords, source_row
                )) {
                continue;
            }
            float distance = squared_spatial_distance(
                query_coords, int(query_row), source_coords, source_row
            );
            int insert_at = selected;
            for (int rank = 0; rank < selected; ++rank) {
                if (distance < local_distances[rank] ||
                    (distance == local_distances[rank] &&
                     source_row < local_sources[rank])) {
                    insert_at = rank;
                    break;
                }
            }
            if (insert_at >= k) {
                continue;
            }
            int last = min(selected, k - 1);
            for (int rank = last; rank > insert_at; --rank) {
                local_distances[rank] = local_distances[rank - 1];
                local_sources[rank] = local_sources[rank - 1];
            }
            local_distances[insert_at] = distance;
            local_sources[insert_at] = source_row;
            selected = min(selected + 1, k);
        }
    }

    int group_base = tid * max_k;
    for (int rank = 0; rank < max_k; ++rank) {
        group_distances[group_base + rank] = local_distances[rank];
        group_sources[group_base + rank] = local_sources[rank];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0 || query_row >= uint(query_count) || max_neighbors <= 0) {
        return;
    }

    int final_selected = 0;
    for (int candidate = 0; candidate < thread_count * max_k; ++candidate) {
        int source_row = group_sources[candidate];
        if (source_row < 0) {
            continue;
        }
        float distance = group_distances[candidate];
        int insert_at = final_selected;
        for (int rank = 0; rank < final_selected; ++rank) {
            int index = slot_start + rank;
            float existing_distance = distances[index];
            int existing_source = source_rows[index];
            if (distance < existing_distance ||
                (distance == existing_distance &&
                 source_row < existing_source)) {
                insert_at = rank;
                break;
            }
        }
        if (insert_at >= k) {
            continue;
        }
        int last = min(final_selected, k - 1);
        for (int rank = last; rank > insert_at; --rank) {
            int dst = slot_start + rank;
            int src = dst - 1;
            source_rows[dst] = source_rows[src];
            distances[dst] = distances[src];
        }
        source_rows[slot_start + insert_at] = source_row;
        distances[slot_start + insert_at] = distance;
        final_selected = min(final_selected + 1, k);
    }

    for (int rank = 0; rank < final_selected; ++rank) {
        int index = slot_start + rank;
        query_rows[index] = int(query_row);
        neighbor_ids[index] = rank;
    }
}

[[kernel]] void compact_neighbor_relation_i32(
    device int* query_rows [[buffer(0)]],
    device int* source_rows [[buffer(1)]],
    device int* neighbor_ids [[buffer(2)]],
    device float* distances [[buffer(3)]],
    device int* row_offsets [[buffer(4)]],
    device int* counts [[buffer(5)]],
    constant const int& query_capacity [[buffer(6)]],
    constant const int& max_neighbors [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_edge = 0;
    int query_count = counts[1];
    for (int query_row = 0; query_row < query_count; ++query_row) {
        row_offsets[query_row] = out_edge;
        int slot_start = query_row * max_neighbors;
        for (int rank = 0; rank < max_neighbors; ++rank) {
            int index = slot_start + rank;
            int source_row = source_rows[index];
            if (source_row < 0) {
                break;
            }
            query_rows[out_edge] = query_row;
            source_rows[out_edge] = source_row;
            neighbor_ids[out_edge] = rank;
            distances[out_edge] = distances[index];
            out_edge += 1;
        }
    }
    row_offsets[query_count] = out_edge;
    counts[0] = out_edge;
    (void)query_capacity;
}

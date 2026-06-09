#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/coords/common.metal"

inline void quantized_point_coord(
    device const float* points,
    device const int* batch_indices,
    int row,
    float voxel_x,
    float voxel_y,
    float voxel_z,
    float origin_x,
    float origin_y,
    float origin_z,
    thread int* out
) {
    int base = row * 3;
    out[0] = batch_indices[row];
    out[1] = int(floor((points[base] - origin_x) / voxel_x));
    out[2] = int(floor((points[base + 1] - origin_y) / voxel_y));
    out[3] = int(floor((points[base + 2] - origin_z) / voxel_z));
}

inline int lookup_quantized_point_hash(
    device const float* points,
    device const int* batch_indices,
    device const int* table_rows,
    int table_capacity,
    thread const int* target,
    float voxel_x,
    float voxel_y,
    float voxel_z,
    float origin_x,
    float origin_y,
    float origin_z
) {
    int slot = coord_hash_i32(target) & (table_capacity - 1);
    for (int probe = 0; probe < table_capacity; ++probe) {
        int row = table_rows[slot];
        if (row < 0) {
            return -1;
        }
        int stored[4];
        quantized_point_coord(
            points,
            batch_indices,
            row,
            voxel_x,
            voxel_y,
            voxel_z,
            origin_x,
            origin_y,
            origin_z,
            stored
        );
        if (target[0] == stored[0] && target[1] == stored[1] &&
            target[2] == stored[2] && target[3] == stored[3]) {
            return row;
        }
        slot = (slot + 1) & (table_capacity - 1);
    }
    return -1;
}

[[kernel]] void clear_sparse_quantization_i32(
    device int* out_coords [[buffer(0)]],
    device int* out_active_rows [[buffer(1)]],
    device int* inverse_rows [[buffer(2)]],
    device int* voxel_counts [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem == 0) {
        out_active_rows[0] = 0;
    }
    if (elem < uint(rows)) {
        inverse_rows[elem] = -1;
        voxel_counts[elem] = 0;
    }
    if (elem < uint(rows * 4)) {
        out_coords[elem] = 0;
    }
}

[[kernel]] void build_quantized_point_hash_i32(
    device const float* points [[buffer(0)]],
    device const int* batch_indices [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device atomic_int* table_rows [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& table_capacity [[buffer(5)]],
    constant const float& voxel_x [[buffer(6)]],
    constant const float& voxel_y [[buffer(7)]],
    constant const float& voxel_z [[buffer(8)]],
    constant const float& origin_x [[buffer(9)]],
    constant const float& origin_y [[buffer(10)]],
    constant const float& origin_z [[buffer(11)]],
    uint row [[thread_position_in_grid]]
) {
    int point_count = min(active_rows[0], rows);
    if (row >= uint(point_count)) {
        return;
    }
    int candidate[4];
    quantized_point_coord(
        points,
        batch_indices,
        int(row),
        voxel_x,
        voxel_y,
        voxel_z,
        origin_x,
        origin_y,
        origin_z,
        candidate
    );
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
        quantized_point_coord(
            points,
            batch_indices,
            expected,
            voxel_x,
            voxel_y,
            voxel_z,
            origin_x,
            origin_y,
            origin_z,
            stored
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

[[kernel]] void plan_quantized_points_i32(
    device const float* points [[buffer(0)]],
    device const int* batch_indices [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const int* table_rows [[buffer(3)]],
    device int* selected [[buffer(4)]],
    constant const int& rows [[buffer(5)]],
    constant const int& table_capacity [[buffer(6)]],
    constant const float& voxel_x [[buffer(7)]],
    constant const float& voxel_y [[buffer(8)]],
    constant const float& voxel_z [[buffer(9)]],
    constant const float& origin_x [[buffer(10)]],
    constant const float& origin_y [[buffer(11)]],
    constant const float& origin_z [[buffer(12)]],
    uint row [[thread_position_in_grid]]
) {
    int point_count = min(active_rows[0], rows);
    if (row >= uint(rows)) {
        return;
    }
    if (row >= uint(point_count)) {
        selected[row] = 0;
        return;
    }
    int candidate[4];
    quantized_point_coord(
        points,
        batch_indices,
        int(row),
        voxel_x,
        voxel_y,
        voxel_z,
        origin_x,
        origin_y,
        origin_z,
        candidate
    );
    selected[row] = lookup_quantized_point_hash(
                        points,
                        batch_indices,
                        table_rows,
                        table_capacity,
                        candidate,
                        voxel_x,
                        voxel_y,
                        voxel_z,
                        origin_x,
                        origin_y,
                        origin_z
                    ) == int(row);
}

[[kernel]] void prefix_quantized_points_i32(
    device const int* active_rows [[buffer(0)]],
    device const int* selected [[buffer(1)]],
    device int* out_active_rows [[buffer(2)]],
    device int* representative_voxels [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    int point_count = min(active_rows[0], rows);
    int out_count = 0;
    for (int row = 0; row < point_count; ++row) {
        if (selected[row] != 0) {
            representative_voxels[row] = out_count++;
            continue;
        }
        representative_voxels[row] = -1;
    }
    for (int row = point_count; row < rows; ++row) {
        representative_voxels[row] = -1;
    }
    out_active_rows[0] = out_count;
}

[[kernel]] void fill_quantized_points_i32(
    device const float* points [[buffer(0)]],
    device const int* batch_indices [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const int* selected [[buffer(3)]],
    device const int* representative_voxels [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const float& voxel_x [[buffer(7)]],
    constant const float& voxel_y [[buffer(8)]],
    constant const float& voxel_z [[buffer(9)]],
    constant const float& origin_x [[buffer(10)]],
    constant const float& origin_y [[buffer(11)]],
    constant const float& origin_z [[buffer(12)]],
    uint row [[thread_position_in_grid]]
) {
    int point_count = min(active_rows[0], rows);
    if (row >= uint(point_count) || selected[row] == 0) {
        return;
    }

    int candidate[4];
    quantized_point_coord(
        points,
        batch_indices,
        int(row),
        voxel_x,
        voxel_y,
        voxel_z,
        origin_x,
        origin_y,
        origin_z,
        candidate
    );
    write_coord(out_coords, representative_voxels[row], candidate);
}

[[kernel]] void map_quantized_points_i32(
    device const float* points [[buffer(0)]],
    device const int* batch_indices [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device const int* table_rows [[buffer(3)]],
    device const int* representative_voxels [[buffer(4)]],
    device int* inverse_rows [[buffer(5)]],
    device atomic_int* voxel_counts [[buffer(6)]],
    constant const int& rows [[buffer(7)]],
    constant const int& table_capacity [[buffer(8)]],
    constant const float& voxel_x [[buffer(9)]],
    constant const float& voxel_y [[buffer(10)]],
    constant const float& voxel_z [[buffer(11)]],
    constant const float& origin_x [[buffer(12)]],
    constant const float& origin_y [[buffer(13)]],
    constant const float& origin_z [[buffer(14)]],
    uint row [[thread_position_in_grid]]
) {
    int point_count = min(active_rows[0], rows);
    if (row >= uint(point_count)) {
        return;
    }
    int candidate[4];
    quantized_point_coord(
        points,
        batch_indices,
        int(row),
        voxel_x,
        voxel_y,
        voxel_z,
        origin_x,
        origin_y,
        origin_z,
        candidate
    );
    int representative = lookup_quantized_point_hash(
        points,
        batch_indices,
        table_rows,
        table_capacity,
        candidate,
        voxel_x,
        voxel_y,
        voxel_z,
        origin_x,
        origin_y,
        origin_z
    );
    int voxel_row = representative_voxels[representative];
    inverse_rows[row] = voxel_row;
    atomic_fetch_add_explicit(
        &voxel_counts[voxel_row], 1, memory_order_relaxed
    );
}

inline float
voxel_reduce_scale(int reduce, device const int* voxel_counts, int voxel_row) {
    if (reduce == 1) {
        return 1.0f / float(max(voxel_counts[voxel_row], 1));
    }
    return 1.0f;
}

[[kernel]] void clear_voxelized_features_f32(
    device float* out [[buffer(0)]],
    constant const int& elements [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(elements)) {
        out[elem] = 0.0f;
    }
}

[[kernel]] void scatter_voxelized_features_f32_i32(
    device const float* feats [[buffer(0)]],
    device const int* inverse_rows [[buffer(1)]],
    device const int* voxel_counts [[buffer(2)]],
    device const int* active_rows [[buffer(3)]],
    device atomic_float* out [[buffer(4)]],
    constant const int& reduce [[buffer(5)]],
    constant const int& point_rows [[buffer(6)]],
    constant const int& voxel_rows [[buffer(7)]],
    constant const int& channels [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = point_rows * channels;
    if (elem >= uint(total)) {
        return;
    }
    int point_row = int(elem) / channels;
    int channel = int(elem) - point_row * channels;
    if (point_row >= min(active_rows[0], point_rows)) {
        return;
    }
    int voxel_row = inverse_rows[point_row];
    if (voxel_row < 0 || voxel_row >= voxel_rows) {
        return;
    }
    float scale = voxel_reduce_scale(reduce, voxel_counts, voxel_row);
    atomic_fetch_add_explicit(
        &out[voxel_row * channels + channel],
        feats[elem] * scale,
        memory_order_relaxed
    );
}

[[kernel]] void voxelize_feature_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const int* inverse_rows [[buffer(1)]],
    device const int* voxel_counts [[buffer(2)]],
    device const int* active_rows [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int& reduce [[buffer(5)]],
    constant const int& point_rows [[buffer(6)]],
    constant const int& voxel_rows [[buffer(7)]],
    constant const int& channels [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = point_rows * channels;
    if (elem >= uint(total)) {
        return;
    }
    int point_row = int(elem) / channels;
    int channel = int(elem) - point_row * channels;
    if (point_row >= min(active_rows[0], point_rows)) {
        out[elem] = 0.0f;
        return;
    }
    int voxel_row = inverse_rows[point_row];
    if (voxel_row < 0 || voxel_row >= voxel_rows) {
        out[elem] = 0.0f;
        return;
    }
    float scale = voxel_reduce_scale(reduce, voxel_counts, voxel_row);
    out[elem] = cotangent[voxel_row * channels + channel] * scale;
}

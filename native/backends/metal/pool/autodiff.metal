#include <metal_stdlib>

using namespace metal;

[[kernel]] void sparse_pool_autodiff_clear_f32(
    device float* out [[buffer(0)]],
    constant const int& total [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(total)) {
        out[elem] = 0.0f;
    }
}

inline int pool_edge_rank(
    device const int* in_rows,
    device const int* kernel_ids,
    int edge,
    int n_kernels
) {
    return in_rows[edge] * n_kernels + kernel_ids[edge];
}

[[kernel]] void sparse_pool_relation_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* kernel_ids [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device atomic_float* grad [[buffer(8)]],
    constant const int& reduce [[buffer(9)]],
    constant const int& in_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& channels [[buffer(13)]],
    constant const int& cotangent_s0 [[buffer(14)]],
    constant const int& cotangent_s1 [[buffer(15)]],
    constant const int& feat_s0 [[buffer(16)]],
    constant const int& feat_s1 [[buffer(17)]],
    constant const int& pooled_s0 [[buffer(18)]],
    constant const int& pooled_s1 [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    (void)out_rows;
    (void)kernel_ids;
    (void)in_capacity;
    (void)n_kernels;
    int total = out_capacity * channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    if (out_row >= counts[1]) {
        return;
    }

    float pooled_value = pooled[out_row * pooled_s0 + channel * pooled_s1];
    int contributors = row_offsets[out_row + 1] - row_offsets[out_row];
    if (reduce == 1) {
        contributors = 0;
        for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
             ++edge) {
            int in_row = in_rows[edge];
            if (feats[in_row * feat_s0 + channel * feat_s1] == pooled_value) {
                ++contributors;
            }
        }
    }

    float scale = reduce == 0 ? 1.0f : 1.0f / float(max(contributors, 1));
    float contribution =
        cotangent[out_row * cotangent_s0 + channel * cotangent_s1] * scale;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        int in_row = in_rows[edge];
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

[[kernel]] void sparse_pool_relation_sum_avg_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* kernel_ids [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device atomic_float* grad [[buffer(8)]],
    constant const int& reduce [[buffer(9)]],
    constant const int& in_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& channels [[buffer(13)]],
    constant const int& cotangent_s0 [[buffer(14)]],
    constant const int& cotangent_s1 [[buffer(15)]],
    constant const int& feat_s0 [[buffer(16)]],
    constant const int& feat_s1 [[buffer(17)]],
    constant const int& pooled_s0 [[buffer(18)]],
    constant const int& pooled_s1 [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    (void)feats;
    (void)kernel_ids;
    (void)in_capacity;
    (void)n_kernels;
    (void)feat_s0;
    (void)feat_s1;
    (void)pooled;
    (void)pooled_s0;
    (void)pooled_s1;
    int edge_count = counts[0];
    int total = edge_count * channels;
    if (elem >= uint(total)) {
        return;
    }

    int edge = int(elem) / channels;
    int channel = int(elem) - edge * channels;
    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity) {
        return;
    }

    int degree = row_offsets[out_row + 1] - row_offsets[out_row];
    float scale = reduce == 2 ? 1.0f / float(max(degree, 1)) : 1.0f;
    float contribution =
        cotangent[out_row * cotangent_s0 + channel * cotangent_s1] * scale;
    atomic_fetch_add_explicit(
        &grad[in_row * channels + channel], contribution, memory_order_relaxed
    );
}

[[kernel]] void sparse_pool_relation_jvp_f32_i32(
    device const float* tangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* kernel_ids [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant const int& reduce [[buffer(9)]],
    constant const int& in_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& channels [[buffer(13)]],
    constant const int& tangent_s0 [[buffer(14)]],
    constant const int& tangent_s1 [[buffer(15)]],
    constant const int& feat_s0 [[buffer(16)]],
    constant const int& feat_s1 [[buffer(17)]],
    constant const int& pooled_s0 [[buffer(18)]],
    constant const int& pooled_s1 [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    (void)out_rows;
    (void)in_capacity;
    int total = out_capacity * channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    if (out_row >= counts[1]) {
        out[elem] = 0.0f;
        return;
    }

    float pooled_value = pooled[out_row * pooled_s0 + channel * pooled_s1];
    float value = 0.0f;
    int degree = 0;
    int first_rank = in_capacity * n_kernels;
    float first_tangent = 0.0f;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        int in_row = in_rows[edge];
        float tangent_value =
            tangent[in_row * tangent_s0 + channel * tangent_s1];
        if (reduce == 1) {
            if (feats[in_row * feat_s0 + channel * feat_s1] != pooled_value) {
                continue;
            }
            int rank = pool_edge_rank(in_rows, kernel_ids, edge, n_kernels);
            if (rank < first_rank) {
                first_rank = rank;
                first_tangent = tangent_value;
            }
            continue;
        }
        value += tangent_value;
        ++degree;
    }

    if (reduce == 1) {
        value = first_tangent;
    } else if (reduce == 2) {
        value /= float(max(degree, 1));
    }
    out[elem] = value;
}

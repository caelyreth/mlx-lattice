#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/conv/common.metal"

[[kernel]] void sparse_relation_conv_clear_f32(
    device float* out [[buffer(0)]],
    constant const int& total [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(total)) {
        out[elem] = 0.0f;
    }
}

[[kernel]] void sparse_relation_conv_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = out_capacity * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        out[elem] = 0.0f;
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            acc += feats[in_row * feat_s0 + ci * feat_s1] *
                   weights[sparse_conv_weight_offset(
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
    out[elem] = acc;
    (void)out_rows;
}

[[kernel]] void sparse_relation_conv_f32_i32_vec4(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int blocks = out_channels / 4;
    int total = out_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / blocks;
    int co = (int(elem) - out_row * blocks) * 4;
    int out_base = out_row * out_channels + co;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        out[out_base] = 0.0f;
        out[out_base + 1] = 0.0f;
        out[out_base + 2] = 0.0f;
        out[out_base + 3] = 0.0f;
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float4 acc = float4(0.0f);
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * feat_s0 + ci * feat_s1];
            acc += value * float4(
                               weights[sparse_conv_weight_offset(
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
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co + 1,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co + 2,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co + 3,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )]
                           );
        }
    }
    out[out_base] = acc.x;
    out[out_base + 1] = acc.y;
    out[out_base + 2] = acc.z;
    out[out_base + 3] = acc.w;
    (void)out_rows;
}

[[kernel]] void sparse_relation_conv_atomic_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device atomic_float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int edge = int(elem) / out_channels;
    int co = int(elem) - edge * out_channels;
    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }

    float acc = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        acc += feats[in_row * feat_s0 + ci * feat_s1] *
               weights[sparse_conv_weight_offset(
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
    atomic_fetch_add_explicit(
        &out[out_row * out_channels + co], acc, memory_order_relaxed
    );
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device const int* kernel_row_offsets [[buffer(9)]],
    device const int* kernel_edge_ids [[buffer(10)]],
    device float* grad [[buffer(11)]],
    constant const int& edge_capacity [[buffer(12)]],
    constant const int& out_capacity [[buffer(13)]],
    constant const int& in_capacity [[buffer(14)]],
    constant const int& in_channels [[buffer(15)]],
    constant const int& out_channels [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_s0 [[buffer(19)]],
    constant const int& weight_s1 [[buffer(20)]],
    constant const int& weight_s2 [[buffer(21)]],
    constant const int& weight_s3 [[buffer(22)]],
    constant const int& weight_s4 [[buffer(23)]],
    constant const int& weight_layout [[buffer(24)]],
    constant const int& kernel_x [[buffer(25)]],
    constant const int& kernel_y [[buffer(26)]],
    constant const int& kernel_z [[buffer(27)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = in_capacity * in_channels;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / in_channels;
    int ci = int(elem) - in_row * in_channels;
    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            acc += cotangent[out_row * cotangent_s0 + co * cotangent_s1] *
                   weights[sparse_conv_weight_offset(
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
    grad[in_row * in_channels + ci] = acc;
    (void)in_rows;
    (void)row_offsets;
    (void)kernel_row_offsets;
    (void)kernel_edge_ids;
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32_vec4(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device const int* kernel_row_offsets [[buffer(9)]],
    device const int* kernel_edge_ids [[buffer(10)]],
    device float* grad [[buffer(11)]],
    constant const int& edge_capacity [[buffer(12)]],
    constant const int& out_capacity [[buffer(13)]],
    constant const int& in_capacity [[buffer(14)]],
    constant const int& in_channels [[buffer(15)]],
    constant const int& out_channels [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_s0 [[buffer(19)]],
    constant const int& weight_s1 [[buffer(20)]],
    constant const int& weight_s2 [[buffer(21)]],
    constant const int& weight_s3 [[buffer(22)]],
    constant const int& weight_s4 [[buffer(23)]],
    constant const int& weight_layout [[buffer(24)]],
    constant const int& kernel_x [[buffer(25)]],
    constant const int& kernel_y [[buffer(26)]],
    constant const int& kernel_z [[buffer(27)]],
    uint elem [[thread_position_in_grid]]
) {
    int blocks = in_channels / 4;
    int total = in_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / blocks;
    int ci = (int(elem) - in_row * blocks) * 4;
    int edge_count = min(counts[0], edge_capacity);
    float4 acc = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            float value = cotangent[out_row * cotangent_s0 + co * cotangent_s1];
            acc += value * float4(
                               weights[sparse_conv_weight_offset(
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
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 1,
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
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 2,
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
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 3,
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
                               )]
                           );
        }
    }
    int grad_base = in_row * in_channels + ci;
    grad[grad_base] = acc.x;
    grad[grad_base + 1] = acc.y;
    grad[grad_base + 2] = acc.z;
    grad[grad_base + 3] = acc.w;
    (void)in_rows;
    (void)row_offsets;
    (void)kernel_row_offsets;
    (void)kernel_edge_ids;
}

[[kernel]] void sparse_relation_conv_weight_grad_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device const int* kernel_row_offsets [[buffer(9)]],
    device const int* kernel_edge_ids [[buffer(10)]],
    device float* grad [[buffer(11)]],
    constant const int& edge_capacity [[buffer(12)]],
    constant const int& out_capacity [[buffer(13)]],
    constant const int& n_kernels [[buffer(14)]],
    constant const int& in_channels [[buffer(15)]],
    constant const int& out_channels [[buffer(16)]],
    constant const int& feat_s0 [[buffer(17)]],
    constant const int& feat_s1 [[buffer(18)]],
    constant const int& cotangent_s0 [[buffer(19)]],
    constant const int& cotangent_s1 [[buffer(20)]],
    constant const int& weight_layout [[buffer(21)]],
    constant const int& kernel_x [[buffer(22)]],
    constant const int& kernel_y [[buffer(23)]],
    constant const int& kernel_z [[buffer(24)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = n_kernels * in_channels * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int channel = int(elem) % (in_channels * out_channels);
    int kernel_id = int(elem) / (in_channels * out_channels);
    int ci = channel / out_channels;
    int co = channel - ci * out_channels;

    float acc = 0.0f;
    for (int cursor = kernel_row_offsets[kernel_id];
         cursor < kernel_row_offsets[kernel_id + 1];
         ++cursor) {
        int edge = kernel_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || out_row < 0 || out_row >= out_capacity) {
            continue;
        }
        acc += feats[in_row * feat_s0 + ci * feat_s1] *
               cotangent[out_row * cotangent_s0 + co * cotangent_s1];
    }
    grad[sparse_conv_dense_weight_offset(
        kernel_id,
        ci,
        co,
        weight_layout,
        kernel_x,
        kernel_y,
        kernel_z,
        in_channels,
        out_channels
    )] = acc;
    (void)kernel_ids;
    (void)row_offsets;
    (void)in_row_offsets;
    (void)in_edge_ids;
}

[[kernel]] void sparse_relation_conv_weight_grad_atomic_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device const int* kernel_row_offsets [[buffer(9)]],
    device const int* kernel_edge_ids [[buffer(10)]],
    device atomic_float* grad [[buffer(11)]],
    constant const int& edge_capacity [[buffer(12)]],
    constant const int& out_capacity [[buffer(13)]],
    constant const int& n_kernels [[buffer(14)]],
    constant const int& in_channels [[buffer(15)]],
    constant const int& out_channels [[buffer(16)]],
    constant const int& feat_s0 [[buffer(17)]],
    constant const int& feat_s1 [[buffer(18)]],
    constant const int& cotangent_s0 [[buffer(19)]],
    constant const int& cotangent_s1 [[buffer(20)]],
    constant const int& weight_layout [[buffer(21)]],
    constant const int& kernel_x [[buffer(22)]],
    constant const int& kernel_y [[buffer(23)]],
    constant const int& kernel_z [[buffer(24)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * in_channels * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int channel = int(elem) % (in_channels * out_channels);
    int edge = int(elem) / (in_channels * out_channels);
    int ci = channel / out_channels;
    int co = channel - ci * out_channels;

    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }
    float value = feats[in_row * feat_s0 + ci * feat_s1] *
                  cotangent[out_row * cotangent_s0 + co * cotangent_s1];
    atomic_fetch_add_explicit(
        &grad[sparse_conv_dense_weight_offset(
            kernel_id,
            ci,
            co,
            weight_layout,
            kernel_x,
            kernel_y,
            kernel_z,
            in_channels,
            out_channels
        )],
        value,
        memory_order_relaxed
    );
    (void)row_offsets;
    (void)in_row_offsets;
    (void)in_edge_ids;
    (void)kernel_row_offsets;
    (void)kernel_edge_ids;
    (void)n_kernels;
}

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
    device atomic_float* out [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& out_capacity [[buffer(8)]],
    constant const int& in_channels [[buffer(9)]],
    constant const int& out_channels [[buffer(10)]],
    constant const int& feat_s0 [[buffer(11)]],
    constant const int& feat_s1 [[buffer(12)]],
    constant const int& weight_s0 [[buffer(13)]],
    constant const int& weight_s1 [[buffer(14)]],
    constant const int& weight_s2 [[buffer(15)]],
    constant const int& weight_s3 [[buffer(16)]],
    constant const int& weight_s4 [[buffer(17)]],
    constant const int& weight_layout [[buffer(18)]],
    constant const int& kernel_x [[buffer(19)]],
    constant const int& kernel_y [[buffer(20)]],
    constant const int& kernel_z [[buffer(21)]],
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
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device atomic_float* grad [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& out_capacity [[buffer(8)]],
    constant const int& in_channels [[buffer(9)]],
    constant const int& out_channels [[buffer(10)]],
    constant const int& cotangent_s0 [[buffer(11)]],
    constant const int& cotangent_s1 [[buffer(12)]],
    constant const int& weight_s0 [[buffer(13)]],
    constant const int& weight_s1 [[buffer(14)]],
    constant const int& weight_s2 [[buffer(15)]],
    constant const int& weight_s3 [[buffer(16)]],
    constant const int& weight_s4 [[buffer(17)]],
    constant const int& weight_layout [[buffer(18)]],
    constant const int& kernel_x [[buffer(19)]],
    constant const int& kernel_y [[buffer(20)]],
    constant const int& kernel_z [[buffer(21)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * in_channels;
    if (elem >= uint(total)) {
        return;
    }

    int edge = int(elem) / in_channels;
    int ci = int(elem) - edge * in_channels;
    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }

    float acc = 0.0f;
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
    atomic_fetch_add_explicit(
        &grad[in_row * in_channels + ci], acc, memory_order_relaxed
    );
}

[[kernel]] void sparse_relation_conv_weight_grad_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device atomic_float* grad [[buffer(6)]],
    constant const int& edge_capacity [[buffer(7)]],
    constant const int& out_capacity [[buffer(8)]],
    constant const int& in_channels [[buffer(9)]],
    constant const int& out_channels [[buffer(10)]],
    constant const int& feat_s0 [[buffer(11)]],
    constant const int& feat_s1 [[buffer(12)]],
    constant const int& cotangent_s0 [[buffer(13)]],
    constant const int& cotangent_s1 [[buffer(14)]],
    constant const int& weight_layout [[buffer(15)]],
    constant const int& kernel_x [[buffer(16)]],
    constant const int& kernel_y [[buffer(17)]],
    constant const int& kernel_z [[buffer(18)]],
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
}

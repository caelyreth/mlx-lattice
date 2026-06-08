#include <metal_stdlib>

using namespace metal;

[[kernel]] void spmm_edges_f32_serial(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant const int& edge_count [[buffer(6)]],
    constant const int& in_channels [[buffer(7)]],
    constant const int& out_channels [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_in_rows [[buffer(10)]],
    constant const int& n_kernels [[buffer(11)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * out_channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = 0.0f;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows || kernel_id < 0 || kernel_id >= n_kernels) {
            continue;
        }

        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * in_channels + ci];
            int weight_base = (kernel_id * in_channels + ci) * out_channels;
            int out_base = out_row * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                out[out_base + co] += value * weights[weight_base + co];
            }
        }
    }
}

[[kernel]] void pool_sum_edges_f32_serial(
    device const float* feats [[buffer(0)]],
    device const int* in_rows [[buffer(1)]],
    device const int* out_rows [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant const int& edge_count [[buffer(4)]],
    constant const int& channels [[buffer(5)]],
    constant const int& n_out_rows [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = 0.0f;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int in_base = in_row * channels;
        int out_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            out[out_base + channel] += feats[in_base + channel];
        }
    }
}

[[kernel]] void spmm_edges_input_grad_f32_serial(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device float* grad [[buffer(5)]],
    constant const int& edge_count [[buffer(6)]],
    constant const int& in_channels [[buffer(7)]],
    constant const int& out_channels [[buffer(8)]],
    constant const int& n_in_rows [[buffer(9)]],
    constant const int& n_out_rows [[buffer(10)]],
    constant const int& n_kernels [[buffer(11)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int grad_size = n_in_rows * in_channels;
    for (int index = 0; index < grad_size; ++index) {
        grad[index] = 0.0f;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows || kernel_id < 0 || kernel_id >= n_kernels) {
            continue;
        }

        int grad_base = in_row * in_channels;
        int cotangent_base = out_row * out_channels;
        int weight_base = kernel_id * in_channels * out_channels;
        for (int ci = 0; ci < in_channels; ++ci) {
            int weight_row = weight_base + ci * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                grad[grad_base + ci] +=
                    cotangent[cotangent_base + co] * weights[weight_row + co];
            }
        }
    }
}

[[kernel]] void spmm_edges_weight_grad_f32_serial(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device float* grad [[buffer(5)]],
    constant const int& edge_count [[buffer(6)]],
    constant const int& in_channels [[buffer(7)]],
    constant const int& out_channels [[buffer(8)]],
    constant const int& n_in_rows [[buffer(9)]],
    constant const int& n_out_rows [[buffer(10)]],
    constant const int& n_kernels [[buffer(11)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int grad_size = n_kernels * in_channels * out_channels;
    for (int index = 0; index < grad_size; ++index) {
        grad[index] = 0.0f;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows || kernel_id < 0 || kernel_id >= n_kernels) {
            continue;
        }

        int feat_base = in_row * in_channels;
        int cotangent_base = out_row * out_channels;
        int grad_base = kernel_id * in_channels * out_channels;
        for (int ci = 0; ci < in_channels; ++ci) {
            int grad_row = grad_base + ci * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                grad[grad_row + co] +=
                    feats[feat_base + ci] * cotangent[cotangent_base + co];
            }
        }
    }
}

[[kernel]] void pool_max_edges_f32_serial(
    device const float* feats [[buffer(0)]],
    device const int* in_rows [[buffer(1)]],
    device const int* out_rows [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant const int& edge_count [[buffer(4)]],
    constant const int& channels [[buffer(5)]],
    constant const int& n_out_rows [[buffer(6)]],
    constant const int& n_in_rows [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = -INFINITY;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int in_base = in_row * channels;
        int out_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            int index = out_base + channel;
            out[index] = max(out[index], feats[in_base + channel]);
        }
    }
}

[[kernel]] void pool_sum_edges_grad_f32_serial(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device float* grad [[buffer(5)]],
    constant const int& edge_count [[buffer(6)]],
    constant const int& channels [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }
    (void)feats;
    (void)pooled;

    int grad_size = n_in_rows * channels;
    for (int index = 0; index < grad_size; ++index) {
        grad[index] = 0.0f;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int grad_base = in_row * channels;
        int cotangent_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            grad[grad_base + channel] += cotangent[cotangent_base + channel];
        }
    }
}

[[kernel]] void pool_max_edges_grad_f32_serial(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device float* grad [[buffer(5)]],
    constant const int& edge_count [[buffer(6)]],
    constant const int& channels [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int grad_size = n_in_rows * channels;
    for (int index = 0; index < grad_size; ++index) {
        grad[index] = 0.0f;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int in_base = in_row * channels;
        int out_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            if (feats[in_base + channel] == pooled[out_base + channel]) {
                grad[in_base + channel] += cotangent[out_base + channel];
            }
        }
    }
}

[[kernel]] void pool_max_edges_jvp_f32_serial(
    device const float* tangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant const int& edge_count [[buffer(6)]],
    constant const int& channels [[buffer(7)]],
    constant const int& n_in_rows [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = 0.0f;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows) {
            continue;
        }

        int in_base = in_row * channels;
        int out_base = out_row * channels;
        for (int channel = 0; channel < channels; ++channel) {
            if (feats[in_base + channel] == pooled[out_base + channel]) {
                out[out_base + channel] += tangent[in_base + channel];
            }
        }
    }
}

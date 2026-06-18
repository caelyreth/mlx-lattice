#include "backends/cuda/pool/kernels.cuh"

#include <cuda_runtime.h>

namespace mlx_lattice::backend::cuda::pool {

namespace {

__device__ int elem_1d() { return int(blockIdx.x * blockDim.x + threadIdx.x); }

} // namespace

__global__ void sparse_pool_relation_f32_i32(
    const float* feats,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    float* out,
    int reduce,
    int out_capacity,
    int channels,
    int feat_s0,
    int feat_s1
) {
    (void)out_rows;
    (void)kernel_ids;
    int elem = elem_1d();
    int total = out_capacity * channels;
    if (elem >= total) {
        return;
    }

    int out_row = elem / channels;
    int channel = elem - out_row * channels;
    if (out_row >= counts[1]) {
        out[elem] = reduce == 1 ? -CUDART_INF_F : 0.0f;
        return;
    }

    float acc = reduce == 1 ? -CUDART_INF_F : 0.0f;
    int degree = 0;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        int in_row = in_rows[edge];
        float value = feats[in_row * feat_s0 + channel * feat_s1];
        acc = reduce == 1 ? fmaxf(acc, value) : acc + value;
        ++degree;
    }
    out[elem] = reduce == 2 ? acc / float(max(degree, 1)) : acc;
}

__global__ void sparse_pool_relation_sum_avg_input_grad_f32_i32(
    const float* cotangent,
    const float* feats,
    const float* pooled,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    const int* in_row_offsets,
    const int* in_edge_ids,
    float* out,
    int reduce,
    int in_capacity,
    int out_capacity,
    int n_kernels,
    int channels,
    int cot_s0,
    int cot_s1,
    int feat_s0,
    int feat_s1,
    int pooled_s0,
    int pooled_s1
) {
    (void)feats;
    (void)pooled;
    (void)in_rows;
    (void)kernel_ids;
    (void)out_capacity;
    (void)n_kernels;
    (void)feat_s0;
    (void)feat_s1;
    (void)pooled_s0;
    (void)pooled_s1;
    int elem = elem_1d();
    int total = in_capacity * channels;
    if (elem >= total) {
        return;
    }
    int in_row = elem / channels;
    int channel = elem - in_row * channels;
    float acc = 0.0f;
    for (int item = in_row_offsets[in_row]; item < in_row_offsets[in_row + 1];
         ++item) {
        int edge = in_edge_ids[item];
        int out_row = out_rows[edge];
        if (out_row < 0 || out_row >= counts[1]) {
            continue;
        }
        float scale = 1.0f;
        if (reduce == 2) {
            int degree = row_offsets[out_row + 1] - row_offsets[out_row];
            scale = 1.0f / float(max(degree, 1));
        }
        acc += cotangent[out_row * cot_s0 + channel * cot_s1] * scale;
    }
    out[in_row * channels + channel] = acc;
}

__global__ void sparse_pool_relation_max_input_grad_f32_i32(
    const float* cotangent,
    const float* feats,
    const float* pooled,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    const int* in_row_offsets,
    const int* in_edge_ids,
    float* out,
    int reduce,
    int in_capacity,
    int out_capacity,
    int n_kernels,
    int channels,
    int cot_s0,
    int cot_s1,
    int feat_s0,
    int feat_s1,
    int pooled_s0,
    int pooled_s1
) {
    (void)in_rows;
    (void)kernel_ids;
    (void)row_offsets;
    (void)reduce;
    (void)out_capacity;
    (void)n_kernels;
    int elem = elem_1d();
    int total = in_capacity * channels;
    if (elem >= total) {
        return;
    }
    int in_row = elem / channels;
    int channel = elem - in_row * channels;
    float value = feats[in_row * feat_s0 + channel * feat_s1];
    float acc = 0.0f;
    for (int item = in_row_offsets[in_row]; item < in_row_offsets[in_row + 1];
         ++item) {
        int edge = in_edge_ids[item];
        int out_row = out_rows[edge];
        if (out_row < 0 || out_row >= counts[1]) {
            continue;
        }
        float pooled_value = pooled[out_row * pooled_s0 + channel * pooled_s1];
        acc += value == pooled_value
                   ? cotangent[out_row * cot_s0 + channel * cot_s1]
                   : 0.0f;
    }
    out[in_row * channels + channel] = acc;
}

__global__ void sparse_pool_relation_exclusive_input_grad_f32_i32(
    const float* cotangent,
    const float* feats,
    const float* pooled,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    const int* in_row_offsets,
    const int* in_edge_ids,
    float* out,
    int reduce,
    int in_capacity,
    int out_capacity,
    int n_kernels,
    int channels,
    int cot_s0,
    int cot_s1,
    int feat_s0,
    int feat_s1,
    int pooled_s0,
    int pooled_s1
) {
    (void)feats;
    (void)pooled;
    (void)in_rows;
    (void)kernel_ids;
    (void)in_row_offsets;
    (void)out_capacity;
    (void)n_kernels;
    (void)feat_s0;
    (void)feat_s1;
    (void)pooled_s0;
    (void)pooled_s1;
    int elem = elem_1d();
    int total = in_capacity * channels;
    if (elem >= total) {
        return;
    }
    int in_row = elem / channels;
    int channel = elem - in_row * channels;
    int edge = in_edge_ids[in_row];
    int out_row = edge >= 0 ? out_rows[edge] : -1;
    float acc = 0.0f;
    if (out_row >= 0 && out_row < counts[1]) {
        float scale = 1.0f;
        if (reduce == 2) {
            int degree = row_offsets[out_row + 1] - row_offsets[out_row];
            scale = 1.0f / float(max(degree, 1));
        }
        acc = cotangent[out_row * cot_s0 + channel * cot_s1] * scale;
    }
    out[in_row * channels + channel] = acc;
}

__global__ void sparse_pool_relation_jvp_f32_i32(
    const float* tangent,
    const float* feats,
    const float* pooled,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    float* out,
    int reduce,
    int in_capacity,
    int out_capacity,
    int n_kernels,
    int channels,
    int tan_s0,
    int tan_s1,
    int feat_s0,
    int feat_s1,
    int pooled_s0,
    int pooled_s1
) {
    (void)out_rows;
    (void)kernel_ids;
    (void)in_capacity;
    (void)n_kernels;
    int elem = elem_1d();
    int total = out_capacity * channels;
    if (elem >= total) {
        return;
    }
    int out_row = elem / channels;
    int channel = elem - out_row * channels;
    if (out_row >= counts[1]) {
        out[elem] = 0.0f;
        return;
    }

    float acc = 0.0f;
    int degree = 0;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        int in_row = in_rows[edge];
        float include = 1.0f;
        if (reduce == 1) {
            float value = feats[in_row * feat_s0 + channel * feat_s1];
            float pooled_value =
                pooled[out_row * pooled_s0 + channel * pooled_s1];
            include = value == pooled_value ? 1.0f : 0.0f;
        }
        acc += tangent[in_row * tan_s0 + channel * tan_s1] * include;
        ++degree;
    }
    out[elem] = reduce == 2 ? acc / float(max(degree, 1)) : acc;
}

} // namespace mlx_lattice::backend::cuda::pool

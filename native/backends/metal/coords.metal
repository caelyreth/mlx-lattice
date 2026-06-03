#include <metal_stdlib>

using namespace metal;

// MARK: - generative maps

[[kernel]] void build_generative_kernel_map_i32(
    device const int* coords [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device int* in_rows [[buffer(2)]],
    device int* out_rows [[buffer(3)]],
    device int* kernel_ids [[buffer(4)]],
    device int* out_coords [[buffer(5)]],
    device int* output_csr_offsets [[buffer(6)]],
    device int* output_csr_in_rows [[buffer(7)]],
    device int* output_csr_kernel_ids [[buffer(8)]],
    device int* kernel_bucket_offsets [[buffer(9)]],
    device int* kernel_bucket_in_rows [[buffer(10)]],
    device int* kernel_bucket_out_rows [[buffer(11)]],
    device int* input_csr_offsets [[buffer(12)]],
    device int* input_csr_out_rows [[buffer(13)]],
    device int* input_csr_kernel_ids [[buffer(14)]],
    constant const int& rows [[buffer(15)]],
    constant const int& kernel_count [[buffer(16)]],
    constant const int& stride_x [[buffer(17)]],
    constant const int& stride_y [[buffer(18)]],
    constant const int& stride_z [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(rows * kernel_count);
    if (elem <= total) {
        output_csr_offsets[elem] = int(elem);
    }
    if (elem <= uint(rows)) {
        input_csr_offsets[elem] = int(elem) * kernel_count;
    }
    if (elem <= uint(kernel_count)) {
        kernel_bucket_offsets[elem] = int(elem) * rows;
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
    int bucket_row = kernel_index * rows + in_row;

    in_rows[out_row] = in_row;
    out_rows[out_row] = out_row;
    kernel_ids[out_row] = kernel_index;
    output_csr_in_rows[out_row] = in_row;
    output_csr_kernel_ids[out_row] = kernel_index;
    kernel_bucket_in_rows[bucket_row] = in_row;
    kernel_bucket_out_rows[bucket_row] = out_row;
    input_csr_out_rows[out_row] = out_row;
    input_csr_kernel_ids[out_row] = kernel_index;
    out_coords[out_base] = coords[in_base];
    out_coords[out_base + 1] =
        coords[in_base + 1] * stride_x + offsets[offset_base];
    out_coords[out_base + 2] =
        coords[in_base + 2] * stride_y + offsets[offset_base + 1];
    out_coords[out_base + 3] =
        coords[in_base + 3] * stride_z + offsets[offset_base + 2];
}

#pragma once

#include <array>
#include <vector>

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

using Triple = std::array<int, 3>;

struct NativeOutputCsrView {
    mx::array offsets;
    mx::array in_rows;
    mx::array kernel_ids;
};

struct NativeKernelBucketView {
    mx::array offsets;
    mx::array in_rows;
    mx::array out_rows;
};

struct NativeInputCsrView {
    mx::array offsets;
    mx::array out_rows;
    mx::array kernel_ids;
};

struct NativeKernelMap {
    mx::array in_rows;
    mx::array out_rows;
    mx::array kernel_ids;
    mx::array out_coords;
    mx::array kernel_offsets;
    NativeOutputCsrView output_csr;
    NativeKernelBucketView kernel_buckets;
    NativeInputCsrView input_csr;
};

std::vector<Triple> kernel_offsets(Triple kernel_size);
std::vector<Triple> kernel_offsets(Triple kernel_size, Triple dilation);

mx::array downsample_coords(const mx::array& coords, Triple stride);
mx::array union_coords(const mx::array& lhs, const mx::array& rhs);
mx::array intersection_coords(const mx::array& lhs, const mx::array& rhs);
mx::array lookup_coords(const mx::array& coords, const mx::array& queries);

NativeKernelMap build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelMap build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
);

NativeKernelMap build_transposed_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

} // namespace mlx_lattice

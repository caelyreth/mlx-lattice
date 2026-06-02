#pragma once

#include "ops/coords.h"

namespace mlx_lattice {

mx::array dispatch_downsample_coords(const mx::array& coords, Triple stride);
mx::array dispatch_union_coords(const mx::array& lhs, const mx::array& rhs);
mx::array
dispatch_intersection_coords(const mx::array& lhs, const mx::array& rhs);
mx::array
dispatch_lookup_coords(const mx::array& coords, const mx::array& queries);

NativeKernelMap dispatch_build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelMap dispatch_build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
);

NativeKernelMap dispatch_build_transposed_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

} // namespace mlx_lattice

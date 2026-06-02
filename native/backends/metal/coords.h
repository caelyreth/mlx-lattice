#pragma once

#include "ops/coords.h"

namespace mlx_lattice::metal {

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

} // namespace mlx_lattice::metal

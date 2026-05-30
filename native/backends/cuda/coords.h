#pragma once

#include "ops/coords.h"

namespace mlx_lattice::cuda {

KernelMapData
build_subm_kernel_map(const mx::array& coords, Triple kernel_size);

KernelMapData build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
);

} // namespace mlx_lattice::cuda

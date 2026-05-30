#pragma once

#include "ops/coords.h"

namespace mlx_lattice::metal {

KernelMapData
build_subm_kernel_map(const mx::array& coords, Triple kernel_size);

} // namespace mlx_lattice::metal

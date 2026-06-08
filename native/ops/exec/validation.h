#pragma once

#include "ops/exec.h"

namespace mlx_lattice {

void validate_sparse_conv(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights
);

void validate_sparse_pool(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats
);

} // namespace mlx_lattice

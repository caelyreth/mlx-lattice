#pragma once

#include "ops/exec.h"

namespace mlx_lattice {

void validate_sparse_pool_features(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    const mx::array& in_row_offsets,
    const mx::array& in_edge_ids,
    int out_capacity,
    int n_kernels
);

} // namespace mlx_lattice

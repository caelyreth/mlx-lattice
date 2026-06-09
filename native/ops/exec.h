#pragma once

#include "ops/exec/types.h"

namespace mlx_lattice {

namespace mx = mlx::core;

mx::array sparse_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets,
    const mx::array& in_row_offsets,
    const mx::array& in_edge_ids,
    const mx::array& kernel_row_offsets,
    const mx::array& kernel_edge_ids,
    int out_capacity,
    int n_kernels
);

mx::array sparse_pool_features(
    PoolReduceOp op,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    const mx::array& in_row_offsets,
    const mx::array& in_edge_ids,
    int out_capacity,
    int n_kernels,
    PoolInputLayout input_layout
);

} // namespace mlx_lattice

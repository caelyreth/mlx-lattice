#pragma once

#include "ops/exec.h"

namespace mlx_lattice {

mx::array make_sparse_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const SparseRelationExecutionViews& views
);

mx::array make_sparse_pool_features(
    PoolReduceOp reduce,
    const mx::array& feats,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const SparseRelationCSRView& output_csr,
    PoolInputLayout input_layout
);

} // namespace mlx_lattice

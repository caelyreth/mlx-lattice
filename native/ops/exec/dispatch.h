#pragma once

#include "ops/exec.h"
#include "ops/exec/types.h"

namespace mlx_lattice {

NativeSparseTensorOutput dispatch_sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding
);

NativeSparseTensorOutput dispatch_sparse_pool(
    PoolReduceOp reduce,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets,
    Triple stride,
    Triple padding
);

} // namespace mlx_lattice

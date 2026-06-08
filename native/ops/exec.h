#pragma once

#include "mlx/array.h"
#include "ops/exec/types.h"

namespace mlx_lattice {

namespace mx = mlx::core;

NativeSparseTensorOutput sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeSparseTensorOutput sparse_pool(
    PoolReduceOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

} // namespace mlx_lattice

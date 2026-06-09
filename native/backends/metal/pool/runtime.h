#pragma once

#include <vector>

#include "mlx/stream.h"
#include "ops/exec/types.h"

namespace mlx_lattice::backend::metal::pool {

bool is_supported(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets
);

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

} // namespace mlx_lattice::backend::metal::pool

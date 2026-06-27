#pragma once

#include <vector>

#include "features/convolution/contract.h"
#include "mlx/stream.h"

namespace mlx_lattice::backend::metal::tensor_ops::conv::sorted_igemm {

bool supports(SparseConvShape shape, const std::vector<mx::array>& inputs);

void encode(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
);

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::sorted_igemm

#include "ops/exec.h"

#include <vector>

#include "ops/coords.h"
#include "ops/exec/factories.h"
#include "ops/exec/validation.h"

namespace mlx_lattice {

namespace {

mx::array make_offsets_array(const std::vector<Triple>& offsets) {
    std::vector<int32_t> flat;
    flat.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        flat.insert(flat.end(), offset.begin(), offset.end());
    }
    return mx::array(
        flat.begin(), mx::Shape{int(offsets.size()), 3}, mx::int32
    );
}

} // namespace

NativeSparseTensorOutput sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    validate_sparse_conv(coords, active_rows, feats, weights);
    auto offsets = op == SparseMapOp::Generative
                       ? kernel_offsets(kernel_size)
                       : kernel_offsets(kernel_size, dilation);
    auto kernel_rows = int(offsets.size());
    if (weights.ndim() == 3 && weights.shape(0) != kernel_rows) {
        throw std::invalid_argument(
            "weight kernel rows must match the sparse convolution kernel."
        );
    }
    if (weights.ndim() == 5 &&
        weights.shape(1) * weights.shape(2) * weights.shape(3) != kernel_rows) {
        throw std::invalid_argument(
            "weight spatial kernel shape must match kernel_size."
        );
    }
    return make_sparse_conv(
        op,
        coords,
        active_rows,
        feats,
        weights,
        make_offsets_array(offsets),
        stride,
        padding
    );
}

NativeSparseTensorOutput sparse_pool(
    PoolReduceOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    validate_sparse_pool(coords, active_rows, feats);
    return make_sparse_pool(
        op,
        coords,
        active_rows,
        feats,
        make_offsets_array(kernel_offsets(kernel_size, dilation)),
        stride,
        padding
    );
}

} // namespace mlx_lattice

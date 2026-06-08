#include "ops/exec/validation.h"

#include <stdexcept>

namespace mlx_lattice {

void validate_sparse_conv(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights
) {
    if (coords.ndim() != 2 || coords.shape(1) != 4) {
        throw std::invalid_argument("coords must have shape (N, 4).");
    }
    if (coords.dtype() != mx::int32 && coords.dtype() != mx::int64) {
        throw std::invalid_argument("coords must be int32 or int64.");
    }
    if (active_rows.shape() != mx::Shape{1} ||
        active_rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            "active_rows must have shape (1,) and int32 dtype."
        );
    }
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, C_in).");
    }
    if (weights.ndim() != 3 && weights.ndim() != 5) {
        throw std::invalid_argument(
            "weights must have shape (K, C_in, C_out) or "
            "(C_out, Kx, Ky, Kz, C_in)."
        );
    }
    if (coords.shape(0) != feats.shape(0)) {
        throw std::invalid_argument(
            "coords and feats must have the same row count."
        );
    }
    if (feats.dtype() != mx::float32 || weights.dtype() != mx::float32) {
        throw std::invalid_argument(
            "sparse_conv currently supports float32 feats and weights."
        );
    }
    auto weight_in_channels =
        weights.ndim() == 3 ? weights.shape(1) : weights.shape(4);
    if (feats.shape(1) != weight_in_channels) {
        throw std::invalid_argument(
            "feats channels must match weights input channels."
        );
    }
}

void validate_sparse_pool(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats
) {
    if (coords.ndim() != 2 || coords.shape(1) != 4) {
        throw std::invalid_argument("coords must have shape (N, 4).");
    }
    if (coords.dtype() != mx::int32 && coords.dtype() != mx::int64) {
        throw std::invalid_argument("coords must be int32 or int64.");
    }
    if (active_rows.shape() != mx::Shape{1} ||
        active_rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            "active_rows must have shape (1,) and int32 dtype."
        );
    }
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, C).");
    }
    if (coords.shape(0) != feats.shape(0)) {
        throw std::invalid_argument(
            "coords and feats must have the same row count."
        );
    }
    if (feats.dtype() != mx::float32) {
        throw std::invalid_argument(
            "sparse_pool currently supports float32 feats."
        );
    }
}

} // namespace mlx_lattice

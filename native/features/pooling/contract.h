#pragma once

#include <cstdint>

#include "features/coordinates/types.h"
#include "foundation/sparse_relation.h"

namespace mlx_lattice {

enum class PoolReduceOp : std::uint8_t {
    Sum,
    Max,
    Avg,
};

enum class PoolInputLayout : std::uint8_t {
    Overlap,
    Exclusive,
};

struct SparsePoolShape {
    int in_capacity;
    int out_capacity;
    int n_kernels;
    int channels;
    bool input_exclusive;
};

inline bool operator==(SparsePoolShape lhs, SparsePoolShape rhs) {
    return lhs.in_capacity == rhs.in_capacity &&
           lhs.out_capacity == rhs.out_capacity &&
           lhs.n_kernels == rhs.n_kernels && lhs.channels == rhs.channels &&
           lhs.input_exclusive == rhs.input_exclusive;
}

inline bool operator!=(SparsePoolShape lhs, SparsePoolShape rhs) {
    return !(lhs == rhs);
}

} // namespace mlx_lattice

#pragma once

#include "ops/coords/types.h"

namespace mlx_lattice {

namespace mx = mlx::core;

enum class PoolReduceOp {
    Sum,
    Max,
    Avg,
};

enum class PoolInputLayout {
    Overlap,
    Exclusive,
};

enum class SparseMapOp {
    Forward,
    Transposed,
    Generative,
};

enum SparseOutputSlot : std::size_t {
    SparseOutFeats = 0,
    SparseOutCoords,
    SparseCounts,
    SparseOutputCount,
};

struct NativeSparseTensorOutput {
    mx::array coords;
    mx::array feats;
    mx::array counts;
};

struct SparseConvShape {
    int in_capacity;
    int out_capacity;
    int n_kernels;
    int in_channels;
    int out_channels;
    int weight_layout;
    int kernel_x;
    int kernel_y;
    int kernel_z;
};

struct SparseConvPlan {
    mx::array in_row_offsets;
    mx::array in_edge_ids;
    mx::array kernel_row_offsets;
    mx::array kernel_edge_ids;
};

struct SparsePoolShape {
    int in_capacity;
    int out_capacity;
    int n_kernels;
    int channels;
    bool input_exclusive;
};

} // namespace mlx_lattice

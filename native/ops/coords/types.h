#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

using Triple = std::array<int, 3>;
using FloatTriple = std::array<float, 3>;

enum class CoordSetOp : std::uint8_t {
    Downsample,
    Union,
    Intersection,
};

enum class CoordRelationOp : std::uint8_t {
    Forward,
    Transposed,
};

enum CoordRelationOutputSlot : std::size_t {
    RelationInRows = 0,
    RelationOutRows,
    RelationKernelIds,
    RelationRowOffsets,
    RelationOutCoords,
    RelationCounts,
    RelationOutputCount,
};

enum class NeighborRelationOp : std::uint8_t {
    Knn,
    Radius,
};

enum class VoxelReduceOp : std::uint8_t {
    Sum,
    Mean,
};

enum NeighborRelationOutputSlot : std::size_t {
    NeighborQueryRows = 0,
    NeighborSourceRows,
    NeighborIds,
    NeighborDistances,
    NeighborCounts,
    NeighborOutputCount,
};

struct NativeKernelRelation {
    // Baseline edge-COO diagnostics plus the first native execution view.
    // Edges are ordered by output row and row_offsets is CSR-style metadata.
    mx::array in_rows;
    mx::array out_rows;
    mx::array kernel_ids;
    mx::array row_offsets;
    mx::array out_coords;
    mx::array counts;
};

struct NativeNeighborRelation {
    mx::array query_rows;
    mx::array source_rows;
    mx::array neighbor_ids;
    mx::array distances;
    mx::array counts;
};

struct NativeCoordSet {
    mx::array coords;
    mx::array count;
};

struct QuantizationSpec {
    FloatTriple voxel_size;
    FloatTriple origin;
};

struct NativeSparseQuantization {
    mx::array coords;
    mx::array active_rows;
    mx::array inverse_rows;
    mx::array counts;
};

struct CoordSetShape {
    int lhs_rows;
    int rhs_rows;
};

struct CoordLookupShape {
    int rows;
    int query_rows;
};

struct NeighborRelationShape {
    int source_rows;
    int query_rows;
    int max_neighbors;
};

struct VoxelFeatureShape {
    int point_rows;
    int voxel_rows;
    int channels;
};

} // namespace mlx_lattice

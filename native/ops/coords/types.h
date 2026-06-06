#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

using Triple = std::array<int, 3>;

enum class CoordSetOp : std::uint8_t {
    Downsample,
    Union,
    Intersection,
};

enum class CoordMapOp : std::uint8_t {
    Forward,
    Transposed,
};

enum CoordMapOutputSlot : std::size_t {
    MapInRows = 0,
    MapOutRows,
    MapKernelIds,
    MapOutCoords,
    MapCounts,
    MapOutputCount,
};

constexpr std::size_t DirectMapOutputCount = MapOutputCount - 1;

struct NativeKernelMap {
    mx::array in_rows;
    mx::array out_rows;
    mx::array kernel_ids;
    mx::array out_coords;
};

struct CoordSetShape {
    int lhs_rows;
    int rhs_rows;
};

struct CoordLookupShape {
    int rows;
    int query_rows;
};

} // namespace mlx_lattice

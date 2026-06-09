#pragma once

#include "ops/coords.h"

namespace mlx_lattice {

NativeCoordSet make_downsample_coords(const mx::array& coords, Triple stride);
NativeCoordSet make_union_coords(const mx::array& lhs, const mx::array& rhs);
NativeCoordSet
make_intersection_coords(const mx::array& lhs, const mx::array& rhs);
mx::array make_lookup_coords(const mx::array& coords, const mx::array& queries);

NativeKernelRelation make_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelRelation make_generative_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride
);

NativeKernelRelation make_transposed_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

} // namespace mlx_lattice

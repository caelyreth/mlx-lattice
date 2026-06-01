from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice._native import pool3d_feats as _pool3d_feats
from mlx_lattice._ops.conv import conv3d
from mlx_lattice.point import kernel_offsets
from mlx_lattice.tensor import SparseTensor
from mlx_lattice.types import triple


def pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> SparseTensor:
    kernel = triple(kernel_size, name='kernel_size')
    op_stride = triple(stride, name='stride')
    mapping = x.kernel_map(kernel_size=kernel, stride=op_stride)

    if mapping.center >= 0:
        volume = len(kernel_offsets(kernel))
        weight = mx.broadcast_to(
            mx.eye(x.channels, dtype=x.feats.dtype),
            (volume, x.channels, x.channels),
        )
        return conv3d(x, weight, kernel_size=kernel, stride=op_stride)

    feats = _pool3d_feats(
        x.feats,
        mapping.residual_maps,
        mapping.residual_kernels,
        mapping.residual_offsets,
        int(mapping.out_coords.shape[0]),
    )
    out_stride = tuple(
        a * b for a, b in zip(x.stride, op_stride, strict=True)
    )
    return SparseTensor(
        mapping.out_coords,
        feats,
        out_stride,
        coord_manager=x.coord_manager,
    )

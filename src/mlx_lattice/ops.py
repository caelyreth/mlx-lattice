from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from mlx_lattice._native import conv3d_feats as _conv3d_feats
from mlx_lattice.point import downsample, kernel_offsets
from mlx_lattice.tensor import SparseTensor
from mlx_lattice.types import triple


def conv3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    dilation: int | Sequence[int] = 1,
    transposed: bool = False,
) -> SparseTensor:
    if transposed:
        raise NotImplementedError(
            'transposed sparse conv is not implemented.'
        )
    if triple(dilation, name='dilation') != (1, 1, 1):
        raise NotImplementedError(
            'sparse conv currently supports dilation=1.'
        )
    if weight.ndim != 3:
        raise ValueError('weight must have shape (K, Cin, Cout).')
    if weight.dtype != mx.float32 or x.feats.dtype != mx.float32:
        raise ValueError('conv3d currently supports float32 tensors.')
    if weight.shape[1] != x.channels:
        raise ValueError(
            'weight input channels must match tensor features.'
        )

    mapping = x.kernel_map(kernel_size=kernel_size, stride=stride)
    if weight.shape[0] != len(mapping.offsets):
        raise ValueError(
            'weight kernel dimension does not match kernel_size.'
        )

    out_rows = int(mapping.out_coords.shape[0])
    feats = _conv3d_feats(
        x.feats,
        weight,
        mapping.maps,
        mapping.kernels,
        out_rows,
    )
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[2]:
            raise ValueError('bias must have shape (Cout,).')
        feats = feats + bias

    in_stride = np.array(x.stride, dtype=np.int64)
    op_stride = np.array(triple(stride, name='stride'), dtype=np.int64)
    out_stride = tuple(int(v) for v in in_stride * op_stride)
    return SparseTensor(mapping.out_coords, feats, out_stride)


def pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> SparseTensor:
    volume = len(kernel_offsets(kernel_size))
    weight = mx.ones((volume, x.channels, x.channels), dtype=x.feats.dtype)
    return conv3d(x, weight, kernel_size=kernel_size, stride=stride)


spdownsample = downsample
sparse_conv3d = conv3d
sparse_pool3d = pool3d

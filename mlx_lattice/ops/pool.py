from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import mlx.core as mx

from mlx_lattice.core import KernelRelation, KernelSpec, SparseTensor
from mlx_lattice.core.types import Triple
from mlx_lattice.ops.exec import execute_pool_max, execute_pool_sum
from mlx_lattice.ops.maps import kernel_relation

PoolMode = Literal['sum', 'max', 'avg']

__all__ = [
    'avg_pool3d',
    'global_avg_pool',
    'global_max_pool',
    'global_sum_pool',
    'max_pool3d',
    'pool3d',
    'sum_pool3d',
]


def pool3d(
    x: SparseTensor,
    *,
    mode: PoolMode = 'sum',
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    relation = kernel_relation(
        x,
        kernel_size=spec.size,
        stride=spec.stride,
        padding=spec.padding,
        dilation=spec.dilation,
    )
    feats = _pool_features(x, relation, mode)
    return _pooled_tensor(
        x,
        relation,
        feats,
        output_stride=_mul_stride(x.stride, spec.stride),
    )


def sum_pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    return pool3d(
        x,
        mode='sum',
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def max_pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    return pool3d(
        x,
        mode='max',
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def avg_pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    return pool3d(
        x,
        mode='avg',
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def global_sum_pool(x: SparseTensor) -> mx.array:
    return _stack_batch_reductions(x, mode='sum')


def global_avg_pool(x: SparseTensor) -> mx.array:
    return _stack_batch_reductions(x, mode='avg')


def global_max_pool(x: SparseTensor) -> mx.array:
    return _stack_batch_reductions(x, mode='max')


# MARK: - local pooling


def _pool_features(
    x: SparseTensor,
    relation: KernelRelation,
    mode: PoolMode,
) -> mx.array:
    _validate_pool_dtype(x.feats)
    if mode == 'sum':
        return execute_pool_sum(x.feats, relation)
    if mode == 'max':
        return execute_pool_max(x.feats, relation)
    if mode == 'avg':
        sums = execute_pool_sum(x.feats, relation)
        counts = execute_pool_sum(_ones_like_rows(x), relation)
        return sums / counts
    raise ValueError("mode must be 'sum', 'max', or 'avg'.")


def _pooled_tensor(
    x: SparseTensor,
    relation: KernelRelation,
    feats: mx.array,
    *,
    output_stride: Triple,
) -> SparseTensor:
    if relation.out_coords is None:
        raise ValueError(
            'pooling relations must define output coordinates.'
        )
    return SparseTensor(
        relation.out_coords,
        feats,
        stride=output_stride,
        coord_manager=x.coord_manager,
    )


def _ones_like_rows(x: SparseTensor) -> mx.array:
    return mx.ones((x.n_points, 1), dtype=x.feats.dtype)


# MARK: - global pooling


def _stack_batch_reductions(x: SparseTensor, *, mode: PoolMode) -> mx.array:
    _validate_pool_dtype(x.feats)
    values = []
    for rows in x.batch_rows:
        if int(rows.shape[0]) == 0:
            values.append(_empty_global_value(x, mode=mode))
            continue
        feats = mx.take(x.feats, rows, axis=0)
        if mode == 'sum':
            values.append(mx.sum(feats, axis=0))
        elif mode == 'max':
            values.append(mx.max(feats, axis=0))
        elif mode == 'avg':
            values.append(mx.sum(feats, axis=0) / int(rows.shape[0]))
        else:
            raise ValueError("mode must be 'sum', 'max', or 'avg'.")
    if not values:
        return mx.zeros((0, x.channels), dtype=x.feats.dtype)
    return mx.stack(values, axis=0)


def _empty_global_value(x: SparseTensor, *, mode: PoolMode) -> mx.array:
    if mode == 'max':
        raise ValueError('global_max_pool does not support empty batches.')
    return mx.zeros((x.channels,), dtype=x.feats.dtype)


# MARK: - validation


def _validate_pool_dtype(feats: mx.array) -> None:
    if feats.dtype != mx.float32:
        raise ValueError('pooling currently supports float32 tensors.')


def _mul_stride(lhs: Triple, rhs: Triple) -> Triple:
    return (lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2])

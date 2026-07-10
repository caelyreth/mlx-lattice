from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import (
    avg_pool3d,
    global_avg_pool,
    global_max_pool,
    global_sum_pool,
    max_pool3d,
    pool_transpose3d,
    sum_pool3d,
    trilinear_upsample3d,
)

from mlx_lattice_bench.cases.common import benchmark_n, param_grid
from mlx_lattice_bench.datasets import SparseArrays, sparse_arrays
from mlx_lattice_bench.harness import BenchmarkCase

type PoolKind = Literal[
    'sum',
    'max',
    'avg',
    'global_sum',
    'global_max',
    'global_avg',
    'transpose_generated',
    'transpose_target',
    'trilinear_generated',
    'trilinear_target',
]


@dataclass(frozen=True, slots=True)
class PoolInputs:
    x: SparseTensor
    coarse: SparseTensor


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None = None,
    channels: tuple[int, ...] | None = None,
) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        dict(item)
        for item in param_grid(
            preset, n_values=n_values, channels=channels or (16, 64)
        )
    )
    return tuple(
        _case(name, kind, params)
        for name, kind in (
            ('sum_pool3d', 'sum'),
            ('max_pool3d', 'max'),
            ('avg_pool3d', 'avg'),
            ('global_sum_pool', 'global_sum'),
            ('global_max_pool', 'global_max'),
            ('global_avg_pool', 'global_avg'),
            ('pool_transpose3d_generated', 'transpose_generated'),
            ('pool_transpose3d_target', 'transpose_target'),
            ('trilinear_upsample3d_generated', 'trilinear_generated'),
            ('trilinear_upsample3d_target', 'trilinear_target'),
        )
    )


def _case(
    name: str,
    kind: PoolKind,
    params: tuple[Mapping[str, Any], ...],
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group='pool',
        params=params,
        setup=_setup,
        prepare=_prepare,
        run=lambda inputs: _run(kind, inputs.x, inputs.coarse),
        compiled=_compiled(kind),
        backward=_backward(kind),
        units=('elements', 'n_in', 'n_out'),
    )


def _setup(params: Mapping[str, Any]) -> SparseArrays:
    return sparse_arrays(
        rows=benchmark_n(params),
        channels=int(params['channels']),
        batches=int(params['batches']),
    )


def _prepare(fixture: SparseArrays) -> PoolInputs:
    x = fixture.tensor()
    return PoolInputs(x, sum_pool3d(x, kernel_size=2, stride=2))


def _run(
    kind: PoolKind, x: SparseTensor, coarse: SparseTensor | None = None
) -> Any:
    if kind == 'sum':
        return sum_pool3d(x, kernel_size=2, stride=2)
    if kind == 'max':
        return max_pool3d(x, kernel_size=2, stride=2)
    if kind == 'avg':
        return avg_pool3d(x, kernel_size=2, stride=2)
    if kind == 'global_sum':
        return global_sum_pool(x)
    if kind == 'global_max':
        return global_max_pool(x)
    if kind == 'transpose_generated':
        if coarse is None:
            raise ValueError('pooling transpose requires a coarse input.')
        return pool_transpose3d(coarse, kernel_size=2, stride=2)
    if kind == 'transpose_target':
        if coarse is None:
            raise ValueError('pooling transpose requires a coarse input.')
        return pool_transpose3d(coarse, x, kernel_size=2, stride=2)
    if kind == 'trilinear_generated':
        if coarse is None:
            raise ValueError(
                'trilinear upsampling requires a coarse input.'
            )
        return trilinear_upsample3d(coarse, stride=2)
    if kind == 'trilinear_target':
        if coarse is None:
            raise ValueError(
                'trilinear upsampling requires a coarse input.'
            )
        return trilinear_upsample3d(coarse, x, stride=2)
    return global_avg_pool(x)


def _compiled(
    kind: PoolKind,
) -> Callable[[SparseArrays], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: SparseArrays) -> tuple[Any, tuple[Any, ...]]:
        base = fixture.tensor()
        coarse = sum_pool3d(base, kernel_size=2, stride=2)

        def fn(feats: mx.array) -> Any:
            x = base.replace(feats=feats)
            current_coarse = coarse.replace(
                feats=sum_pool3d(x, kernel_size=2, stride=2).feats
            )
            out = _run(kind, x, current_coarse)
            return out.feats if isinstance(out, SparseTensor) else out

        return fn, (fixture.feats,)

    return factory


def _backward(
    kind: PoolKind,
) -> Callable[[SparseArrays], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: SparseArrays) -> tuple[Any, tuple[Any, ...]]:
        base = fixture.tensor()
        coarse = sum_pool3d(base, kernel_size=2, stride=2)

        def loss(feats: mx.array) -> mx.array:
            x = base.replace(feats=feats)
            current_coarse = coarse.replace(
                feats=sum_pool3d(x, kernel_size=2, stride=2).feats
            )
            out = _run(kind, x, current_coarse)
            values = out.feats if isinstance(out, SparseTensor) else out
            return mx.sum(values)

        return mx.grad(loss), (fixture.feats,)

    return factory

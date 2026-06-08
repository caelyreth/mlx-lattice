from __future__ import annotations

from collections.abc import Sequence

from mlx_lattice.core.coords.builders import (
    build_generative_relation,
    build_kernel_relation,
    build_transposed_kernel_relation,
    kernel_offsets,
)
from mlx_lattice.core.relations import KernelRelation
from mlx_lattice.core.tensor import SparseTensor

__all__ = [
    'build_generative_relation',
    'build_kernel_relation',
    'build_transposed_kernel_relation',
    'generative_kernel_relation',
    'kernel_offsets',
    'kernel_relation',
    'transposed_kernel_relation',
]


def kernel_relation(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    return x.coord_manager.kernel_relation(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def generative_kernel_relation(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> KernelRelation:
    return x.coord_manager.generative_relation(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
    )


def transposed_kernel_relation(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    return x.coord_manager.transposed_kernel_relation(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

from __future__ import annotations

from collections.abc import Sequence

from mlx_lattice.core.coords.builders import (
    build_generative_map,
    build_kernel_map,
    build_transposed_kernel_map,
    kernel_offsets,
)
from mlx_lattice.core.maps import KernelMap
from mlx_lattice.core.tensor import SparseTensor

__all__ = [
    'build_generative_map',
    'build_kernel_map',
    'build_transposed_kernel_map',
    'generative_kernel_map',
    'kernel_map',
    'kernel_offsets',
    'transposed_kernel_map',
]


def kernel_map(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelMap:
    return x.coord_manager.kernel_map(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def generative_kernel_map(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> KernelMap:
    return x.coord_manager.generative_map(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
    )


def transposed_kernel_map(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelMap:
    return x.coord_manager.transposed_kernel_map(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

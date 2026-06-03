from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import validate_coords
from mlx_lattice.core.maps import (
    InputCsrView,
    KernelBucketView,
    KernelMap,
    KernelSpec,
    OutputCsrView,
)
from mlx_lattice.core.types import Triple, triple

type NativeKernelMap = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    tuple[mx.array, mx.array, mx.array],
    tuple[mx.array, mx.array, mx.array],
    tuple[mx.array, mx.array, mx.array],
]


def kernel_offsets(
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> tuple[Triple, ...]:
    kernel = triple(kernel_size, name='kernel_size')
    rate = triple(dilation, name='dilation')
    _require_positive(kernel, 'kernel_size')
    _require_positive(rate, 'dilation')

    axes = []
    for size in kernel:
        if size % 2 == 1:
            radius = size // 2
            axes.append(range(-radius, radius + 1))
        else:
            axes.append(range(size))

    return tuple(
        (int(x * rate[0]), int(y * rate[1]), int(z * rate[2]))
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    )


def build_kernel_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelMap:
    validate_coords(coords)
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return _kernel_map_from_native(
        ext.build_kernel_map(
            coords,
            spec.size,
            spec.stride,
            spec.padding,
            spec.dilation,
        ),
        n_in_rows=int(coords.shape[0]),
    )


def build_generative_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> KernelMap:
    validate_coords(coords)
    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    _require_positive(kernel, 'kernel_size')
    _require_positive(step, 'stride')

    return _kernel_map_from_native(
        ext.build_generative_map(
            coords,
            kernel,
            step,
        ),
        n_in_rows=int(coords.shape[0]),
    )


def build_transposed_kernel_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelMap:
    validate_coords(coords)
    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    pad = triple(padding, name='padding')
    rate = triple(dilation, name='dilation')
    _require_positive(kernel, 'kernel_size')
    _require_positive(step, 'stride')
    _require_nonnegative(pad, 'padding')
    _require_positive(rate, 'dilation')

    return _kernel_map_from_native(
        ext.build_transposed_kernel_map(
            coords,
            kernel,
            step,
            pad,
            rate,
        ),
        n_in_rows=int(coords.shape[0]),
    )


# MARK: - views


def _kernel_map_from_native(
    native: NativeKernelMap,
    *,
    n_in_rows: int,
) -> KernelMap:
    (
        in_rows,
        out_rows,
        kernel_ids,
        out_coords,
        offset_values,
        output_csr,
        kernel_buckets,
        input_csr,
    ) = native
    offsets = _offsets_from_array(offset_values)
    return KernelMap(
        in_rows,
        out_rows,
        kernel_ids,
        kernel_offsets=offsets,
        out_coords=out_coords,
        output_csr=OutputCsrView(*output_csr),
        kernel_buckets=KernelBucketView(*kernel_buckets),
        input_csr=InputCsrView(*input_csr),
        n_in_rows=n_in_rows,
        n_out_rows=int(out_coords.shape[0]),
        n_kernels=len(offsets),
    )


# MARK: - arrays


def _offsets_from_array(values: mx.array) -> tuple[Triple, ...]:
    rows = cast(list[list[int]], values.tolist())
    return tuple((int(row[0]), int(row[1]), int(row[2])) for row in rows)


# MARK: - helpers


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')


def _require_nonnegative(values: Triple, name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f'{name} values must be non-negative.')

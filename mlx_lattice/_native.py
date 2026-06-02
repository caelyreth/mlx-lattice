from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Protocol, cast

import mlx.core as mx

from mlx_lattice.core.types import Triple

type NativeKernelMap = tuple[
    mx.array, mx.array, mx.array, mx.array, mx.array
]
type _NativeCoordRows = Sequence[Sequence[int]]
type _NativeRows = Sequence[int]
type _NativeMapData = tuple[
    _NativeRows,
    _NativeRows,
    _NativeRows,
    _NativeCoordRows,
    _NativeCoordRows,
]


class _NativeExtension(Protocol):
    def version(self) -> str: ...
    def capabilities(self) -> dict[str, bool]: ...
    def downsample_coords(
        self,
        coords: _NativeCoordRows,
        dtype_code: int,
        sx: int,
        sy: int,
        sz: int,
    ) -> _NativeCoordRows: ...
    def union_coords(
        self,
        lhs: _NativeCoordRows,
        rhs: _NativeCoordRows,
        dtype_code: int,
    ) -> _NativeCoordRows: ...
    def intersection_coords(
        self,
        lhs: _NativeCoordRows,
        rhs: _NativeCoordRows,
        dtype_code: int,
    ) -> _NativeCoordRows: ...
    def lookup_coords(
        self,
        coords: _NativeCoordRows,
        queries: _NativeCoordRows,
        dtype_code: int,
    ) -> _NativeRows: ...
    def build_kernel_map(
        self,
        coords: _NativeCoordRows,
        dtype_code: int,
        kx: int,
        ky: int,
        kz: int,
        sx: int,
        sy: int,
        sz: int,
        px: int,
        py: int,
        pz: int,
        dx: int,
        dy: int,
        dz: int,
    ) -> _NativeMapData: ...
    def build_generative_map(
        self,
        coords: _NativeCoordRows,
        dtype_code: int,
        kx: int,
        ky: int,
        kz: int,
        sx: int,
        sy: int,
        sz: int,
    ) -> _NativeMapData: ...
    def build_transposed_kernel_map(
        self,
        coords: _NativeCoordRows,
        dtype_code: int,
        kx: int,
        ky: int,
        kz: int,
        sx: int,
        sy: int,
        sz: int,
        px: int,
        py: int,
        pz: int,
        dx: int,
        dy: int,
        dz: int,
    ) -> _NativeMapData: ...


_ext = cast(_NativeExtension, import_module('mlx_lattice._ext'))


def backend_info() -> dict[str, object]:
    return {
        'version': _ext.version(),
        'capabilities': _ext.capabilities(),
    }


def downsample_coords(coords: mx.array, stride: Triple) -> mx.array:
    return _coord_array(
        _ext.downsample_coords(
            _coord_rows(coords),
            _dtype_code(coords),
            *stride,
        ),
        coords.dtype,
    )


def union_coords(lhs: mx.array, rhs: mx.array) -> mx.array:
    return _coord_array(
        _ext.union_coords(
            _coord_rows(lhs),
            _coord_rows(rhs),
            _dtype_code(lhs),
        ),
        lhs.dtype,
    )


def intersection_coords(lhs: mx.array, rhs: mx.array) -> mx.array:
    return _coord_array(
        _ext.intersection_coords(
            _coord_rows(lhs),
            _coord_rows(rhs),
            _dtype_code(lhs),
        ),
        lhs.dtype,
    )


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    return _i32_array(
        _ext.lookup_coords(
            _coord_rows(coords),
            _coord_rows(queries),
            _dtype_code(coords),
        )
    )


def build_kernel_map(
    coords: mx.array,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> NativeKernelMap:
    return _map_data(
        _ext.build_kernel_map(
            _coord_rows(coords),
            _dtype_code(coords),
            *kernel_size,
            *stride,
            *padding,
            *dilation,
        ),
        coords,
    )


def build_generative_map(
    coords: mx.array,
    kernel_size: Triple,
    stride: Triple,
) -> NativeKernelMap:
    return _map_data(
        _ext.build_generative_map(
            _coord_rows(coords),
            _dtype_code(coords),
            *kernel_size,
            *stride,
        ),
        coords,
    )


def build_transposed_kernel_map(
    coords: mx.array,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> NativeKernelMap:
    return _map_data(
        _ext.build_transposed_kernel_map(
            _coord_rows(coords),
            _dtype_code(coords),
            *kernel_size,
            *stride,
            *padding,
            *dilation,
        ),
        coords,
    )


def _coord_array(values: _NativeCoordRows, dtype: mx.Dtype) -> mx.array:
    rows = list(values)
    if not rows:
        return mx.array([], dtype=dtype).reshape((0, 4))
    return mx.array(rows, dtype=dtype)


def _i32_array(values: _NativeRows) -> mx.array:
    return mx.array(list(values), dtype=mx.int32)


def _map_data(values: _NativeMapData, coords: mx.array) -> NativeKernelMap:
    in_rows, out_rows, kernel_ids, out_coords, offsets = values
    return (
        _i32_array(in_rows),
        _i32_array(out_rows),
        _i32_array(kernel_ids),
        _coord_array(out_coords, coords.dtype),
        _coord_array(offsets, mx.int32),
    )


def _coord_rows(coords: mx.array) -> _NativeCoordRows:
    return cast(_NativeCoordRows, coords.tolist())


def _dtype_code(coords: mx.array) -> int:
    if coords.dtype == mx.int32:
        return 32
    if coords.dtype == mx.int64:
        return 64
    raise TypeError('coords must be int32 or int64.')

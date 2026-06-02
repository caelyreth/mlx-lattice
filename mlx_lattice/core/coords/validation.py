from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import mlx.core as mx

Coord = tuple[int, int, int, int]


def validate_coords(coords: mx.array, *, name: str = 'coords') -> None:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f'{name} must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def validate_coord_pair(
    lhs: mx.array,
    rhs: mx.array,
    *,
    lhs_name: str = 'lhs',
    rhs_name: str = 'rhs',
) -> None:
    validate_coords(lhs, name=lhs_name)
    validate_coords(rhs, name=rhs_name)
    if lhs.dtype != rhs.dtype:
        raise ValueError('coordinate arrays must have matching dtype.')


def coord_rows(coords: mx.array) -> list[Coord]:
    values = cast(list[list[int]], coords.tolist())
    return [
        (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        for row in values
    ]


def make_coords_array(values: Iterable[Coord], dtype: mx.Dtype) -> mx.array:
    rows = list(values)
    if not rows:
        return mx.array([], dtype=dtype).reshape((0, 4))
    return mx.array(rows, dtype=dtype)


def make_i32_array(values: Iterable[int]) -> mx.array:
    rows = list(values)
    if not rows:
        return mx.array([], dtype=mx.int32)
    return mx.array(rows, dtype=mx.int32)

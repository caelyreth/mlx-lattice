from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_lattice.types import Triple, triple


@dataclass(frozen=True)
class KernelMap:
    maps: mx.array
    sizes: mx.array
    kernels: mx.array
    out_coords: mx.array
    offsets: tuple[Triple, ...]


def kernel_offsets(kernel_size: int | Sequence[int]) -> tuple[Triple, ...]:
    axes = []
    for size in triple(kernel_size, name='kernel_size'):
        if size <= 0:
            raise ValueError('kernel_size values must be positive.')
        if size % 2:
            radius = size // 2
            axes.append(range(-radius, radius + 1))
        else:
            axes.append(range(size))
    return tuple(
        (int(x), int(y), int(z))
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    )


def downsample(
    coords: mx.array,
    stride: int | Sequence[int] = 2,
) -> mx.array:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError('coords must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError('coords must be int32 or int64.')

    step = np.array(triple(stride, name='stride'), dtype=np.int64)
    if np.any(step <= 0):
        raise ValueError('stride values must be positive.')

    values = np.asarray(coords)
    out = values.copy()
    out[:, 1:] = np.floor_divide(out[:, 1:], step)
    _, keep = np.unique(out, axis=0, return_index=True)
    keep.sort()
    return mx.array(out[keep], dtype=coords.dtype)


def build_kernel_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
) -> KernelMap:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError('coords must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError('coords must be int32 or int64.')

    step = np.array(triple(stride, name='stride'), dtype=np.int64)
    if np.any(step <= 0):
        raise ValueError('stride values must be positive.')

    offsets = kernel_offsets(kernel_size)
    values = np.asarray(coords, dtype=np.int64)
    if values.size == 0:
        return KernelMap(
            maps=mx.array(np.empty((0, 2), dtype=np.int32)),
            sizes=mx.zeros((len(offsets),), dtype=mx.int32),
            kernels=mx.array(np.empty((0,), dtype=np.int32)),
            out_coords=coords,
            offsets=offsets,
        )

    unit_stride = tuple(step.tolist()) == (1, 1, 1)
    out = (
        values.copy()
        if unit_stride
        else np.asarray(downsample(coords, stride))
    )
    offset_values = np.array(offsets, dtype=np.int64)
    mins = _bounds_min(values, out, offset_values, step)
    dims = _bounds_max(values, out, offset_values, step) - mins + 1

    input_keys = _encode(values, mins, dims)
    order = np.argsort(input_keys, kind='stable')
    sorted_keys = input_keys[order]
    out_keys = _encode(out, mins, dims)

    maps: list[tuple[int, int]] = []
    kernels: list[int] = []
    sizes: list[int] = []

    for kernel, offset in enumerate(offset_values):
        if unit_stride:
            keys = out_keys + (
                offset[0] * dims[2] * dims[3]
                + offset[1] * dims[3]
                + offset[2]
            )
        else:
            candidates = out.copy()
            candidates[:, 1:] = out[:, 1:] * step + offset
            keys = _encode(candidates, mins, dims)

        positions = np.searchsorted(sorted_keys, keys)
        in_range = positions < sorted_keys.shape[0]
        valid = np.zeros(keys.shape[0], dtype=bool)
        valid[in_range] = sorted_keys[positions[in_range]] == keys[in_range]

        out_rows = np.nonzero(valid)[0].astype(np.int32)
        in_rows = order[positions[valid]].astype(np.int32)
        sizes.append(int(out_rows.shape[0]))
        maps.extend(zip(in_rows.tolist(), out_rows.tolist(), strict=True))
        kernels.extend([kernel] * out_rows.shape[0])

    return KernelMap(
        maps=mx.array(np.array(maps, dtype=np.int32).reshape((-1, 2))),
        sizes=mx.array(np.array(sizes, dtype=np.int32)),
        kernels=mx.array(np.array(kernels, dtype=np.int32)),
        out_coords=mx.array(out, dtype=coords.dtype),
        offsets=offsets,
    )


def _bounds_min(
    coords: np.ndarray,
    out: np.ndarray,
    offsets: np.ndarray,
    stride: np.ndarray,
) -> np.ndarray:
    candidates = out[:, 1:] * stride + offsets.min(axis=0)
    return np.minimum(
        coords.min(axis=0),
        np.concatenate([out[:, :1].min(axis=0), candidates.min(axis=0)]),
    )


def _bounds_max(
    coords: np.ndarray,
    out: np.ndarray,
    offsets: np.ndarray,
    stride: np.ndarray,
) -> np.ndarray:
    candidates = out[:, 1:] * stride + offsets.max(axis=0)
    return np.maximum(
        coords.max(axis=0),
        np.concatenate([out[:, :1].max(axis=0), candidates.max(axis=0)]),
    )


def _encode(
    values: np.ndarray, mins: np.ndarray, dims: np.ndarray
) -> np.ndarray:
    shifted = values.astype(np.int64, copy=False) - mins
    return (
        (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]
    ) * dims[3] + shifted[:, 3]

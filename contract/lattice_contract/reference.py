"""Deterministic, dependency-free reference evaluators for lattice semantics.

These routines are intentionally scalar and unsuitable for production use. They
define compact correctness probes shared by CUDA and Metal tests without
depending on either backend's relation builder or reduction implementation.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from lattice_contract.kernel import centered_kernel_offsets

type Coordinate = tuple[int, int, int, int]

__all__ = ['submanifold_conv3d_f32_to_f64']


def submanifold_conv3d_f32_to_f64(
    coordinates: Sequence[Sequence[int]],
    features: Sequence[Sequence[float]],
    weight: Sequence[Sequence[Sequence[float]]],
    *,
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
    bias: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Evaluate a canonical submanifold convolution from binary32 leaves.

    The operation follows the portable lattice convention: coordinates are
    ``(batch, x, y, z)`` and kernel rows are canonical ``(x, y, z)`` positions
    with ``z`` varying fastest. Inputs are rounded to their binary32 values,
    every product is exactly representable in binary64, and ``math.fsum``
    performs the final correctly rounded binary64 accumulation.

    It exists for conformance tests and diagnostics, not model execution.
    """

    coords = tuple(_coordinate(row) for row in coordinates)
    if len(set(coords)) != len(coords):
        raise ValueError(
            'submanifold reference coordinates must be unique.'
        )
    rows = tuple(tuple(_f32(value) for value in row) for row in features)
    if len(rows) != len(coords):
        raise ValueError('features must have one row per coordinate.')
    if not rows:
        return ()
    in_channels = len(rows[0])
    if any(len(row) != in_channels for row in rows):
        raise ValueError(
            'feature rows must have a consistent channel count.'
        )

    offsets = centered_kernel_offsets(kernel_size, dilation)
    filters = tuple(
        tuple(
            tuple(_f32(value) for value in output) for output in kernel_row
        )
        for kernel_row in weight
    )
    if len(filters) != len(offsets):
        raise ValueError('weight rows must match the kernel volume.')
    if not filters:
        raise ValueError('weight rows must not be empty.')
    if any(len(kernel_row) != in_channels for kernel_row in filters):
        raise ValueError(
            'weight input channels must match feature channels.'
        )
    out_channels = len(filters[0][0])
    if out_channels == 0 or any(
        len(output) != out_channels
        for kernel_row in filters
        for output in kernel_row
    ):
        raise ValueError(
            'weight output channels must be non-empty and consistent.'
        )

    biases = (
        tuple(_f32(value) for value in bias)
        if bias is not None
        else (0.0,) * out_channels
    )
    if len(biases) != out_channels:
        raise ValueError('bias channels must match weight output channels.')

    source_rows = {
        coordinate: index for index, coordinate in enumerate(coords)
    }
    result: list[tuple[float, ...]] = []
    for batch, x, y, z in coords:
        outputs = []
        for out_channel in range(out_channels):
            terms = [biases[out_channel]]
            for kernel_row, (offset_x, offset_y, offset_z) in enumerate(
                offsets
            ):
                source = source_rows.get(
                    (batch, x + offset_x, y + offset_y, z + offset_z)
                )
                if source is None:
                    continue
                terms.extend(
                    rows[source][in_channel]
                    * filters[kernel_row][in_channel][out_channel]
                    for in_channel in range(in_channels)
                )
            outputs.append(math.fsum(terms))
        result.append(tuple(outputs))
    return tuple(result)


def _coordinate(value: Sequence[int]) -> Coordinate:
    if len(value) != 4:
        raise ValueError('coordinates must have (batch, x, y, z) columns.')
    return int(value[0]), int(value[1]), int(value[2]), int(value[3])


def _f32(value: float) -> float:
    return struct.unpack('=f', struct.pack('=f', float(value)))[0]

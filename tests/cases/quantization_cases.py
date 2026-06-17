from __future__ import annotations

from typing import cast

from mlx_lattice.ops import sparse_quantize, voxelize
from tests.cases.types import ValueCase
from tests.support import active_feats, mx


def cases() -> list[ValueCase]:
    return [ValueCase('quantization_and_voxelization', _quantization)]


def _active_rows(values: mx.array, count: mx.array) -> list[int]:
    return cast('list[int]', values[: int(count.tolist()[0])].tolist())


def _active_coords(values: mx.array, count: mx.array) -> list[list[int]]:
    return cast(
        'list[list[int]]', values[: int(count.tolist()[0])].tolist()
    )


def _quantization() -> object:
    points = mx.array(
        [
            [0.2, 0.2, 0.2],
            [0.8, 0.1, 0.1],
            [1.2, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 2.1, 0.0],
        ],
        dtype=mx.float32,
    )
    feats = mx.array(
        [[1.0], [3.0], [5.0], [7.0], [11.0]],
        dtype=mx.float32,
    )
    batches = mx.array([0, 0, 0, 0, 1], dtype=mx.int32)
    quantized = sparse_quantize(points, batch_indices=batches)
    mean = voxelize(points, feats, batch_indices=batches, reduction='mean')
    summed = voxelize(points, feats, batch_indices=batches, reduction='sum')
    mx.eval(
        quantized.coords,
        quantized.active_rows,
        quantized.inverse_rows,
        quantized.counts,
        mean.feats,
        summed.feats,
        mean.active_rows,
    )
    return (
        _active_coords(quantized.coords, quantized.active_count),
        cast('list[int]', quantized.inverse_rows.tolist()),
        _active_rows(quantized.counts, quantized.active_count),
        active_feats(mean).tolist(),
        active_feats(summed).tolist(),
        cast('list[int]', mean.active_rows.tolist()),
    )

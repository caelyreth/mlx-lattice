from __future__ import annotations

from typing import Any, cast

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    avg_pool3d,
    global_avg_pool,
    global_max_pool,
    global_sum_pool,
    max_pool3d,
    pool3d,
    sparse_collate,
    sum_pool3d,
)
from tests.support import mx


def test_local_pooling_uses_kernel_map_edge_reductions() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=mx.float32,
    )
    x = SparseTensor(coords, feats)

    summed = sum_pool3d(x, kernel_size=(3, 1, 1), stride=1)
    maxed = max_pool3d(x, kernel_size=(3, 1, 1), stride=1)
    averaged = avg_pool3d(x, kernel_size=(3, 1, 1), stride=1)

    assert summed.coords.tolist() == coords.tolist()
    assert summed.feats.tolist() == [[3.0, 30.0], [6.0, 60.0], [5.0, 50.0]]
    assert maxed.feats.tolist() == [[2.0, 20.0], [3.0, 30.0], [3.0, 30.0]]
    assert averaged.feats.tolist() == [
        [1.5, 15.0],
        [2.0, 20.0],
        [2.5, 25.0],
    ]


def test_strided_pooling_updates_output_stride_and_manager_context() -> (
    None
):
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0], [4.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = sum_pool3d(x, kernel_size=1, stride=2)

    assert out.coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert out.feats.tolist() == [[1.0], [3.0]]
    assert out.stride == (2, 2, 2)
    assert out.coord_manager is x.coord_manager
    assert out.coord_key != x.coord_key
    assert out.coord_manager.owns(out.coord_key)


def test_global_pooling_reduces_each_batch_independently() -> None:
    x = sparse_collate(
        [
            mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32),
            mx.array([[2, 0, 0], [3, 0, 0]], dtype=mx.int32),
        ],
        [
            mx.array([[1.0, 10.0], [2.0, 20.0]], dtype=mx.float32),
            mx.array([[3.0, 30.0], [5.0, 50.0]], dtype=mx.float32),
        ],
    )

    assert global_sum_pool(x).tolist() == [[3.0, 30.0], [8.0, 80.0]]
    assert global_avg_pool(x).tolist() == [[1.5, 15.0], [4.0, 40.0]]
    assert global_max_pool(x).tolist() == [[2.0, 20.0], [5.0, 50.0]]


def test_pool3d_rejects_invalid_mode_and_dtype() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0]], dtype=mx.int32),
        mx.ones((1, 1), dtype=mx.float32),
    )
    with pytest.raises(ValueError, match='mode'):
        pool3d(x, mode=cast('Any', 'median'))

    half = x.astype(mx.float16)
    with pytest.raises(ValueError, match='float32'):
        sum_pool3d(half)

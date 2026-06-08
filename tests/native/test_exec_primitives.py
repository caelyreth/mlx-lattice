from __future__ import annotations

from typing import cast

from mlx_lattice.core import KernelMap
from mlx_lattice.ops import (
    build_kernel_map,
    pool_max_edges,
    pool_sum_edges,
    spmm_edges,
)
from tests.support import mx, run_with_gpu_default


def _manual_spmm(
    feats: list[list[float]],
    weights: list[list[list[float]]],
    in_rows: list[int],
    out_rows: list[int],
    kernel_ids: list[int],
    n_out_rows: int,
) -> list[list[float]]:
    out_channels = len(weights[0][0])
    out = [[0.0 for _ in range(out_channels)] for _ in range(n_out_rows)]
    for in_row, out_row, kernel_id in zip(
        in_rows, out_rows, kernel_ids, strict=True
    ):
        for ci, value in enumerate(feats[in_row]):
            for co in range(out_channels):
                out[out_row][co] += value * weights[kernel_id][ci][co]
    return out


def test_spmm_edges_matches_manual_reference_with_repeated_outputs() -> (
    None
):
    mapping = KernelMap(
        mx.array([0, 1, 2, 0], dtype=mx.int32),
        mx.array([0, 0, 1, 1], dtype=mx.int32),
        mx.array([0, 1, 0, 1], dtype=mx.int32),
        kernel_offsets=((0, 0, 0), (1, 0, 0)),
        n_in_rows=3,
        n_out_rows=2,
        n_kernels=2,
    )
    feats = mx.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=mx.float32,
    )
    weights = mx.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [1.0, 2.0]],
        ],
        dtype=mx.float32,
    )

    out = spmm_edges(feats, weights, mapping)

    assert out.tolist() == _manual_spmm(
        cast('list[list[float]]', feats.tolist()),
        cast('list[list[list[float]]]', weights.tolist()),
        cast('list[int]', mapping.in_rows.tolist()),
        cast('list[int]', mapping.out_rows.tolist()),
        cast('list[int]', mapping.kernel_ids.tolist()),
        2,
    )


def test_spmm_edges_consumes_lazy_kernel_map_outputs() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    mapping = build_kernel_map(coords, kernel_size=(3, 1, 1))
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weights = mx.ones((3, 1, 1), dtype=mx.float32)

    assert spmm_edges(feats, weights, mapping).tolist() == [
        [3.0],
        [6.0],
        [5.0],
    ]


def test_pool_edge_reductions_match_manual_reference() -> None:
    mapping = KernelMap(
        mx.array([0, 1, 2, 0], dtype=mx.int32),
        mx.array([0, 0, 1, 1], dtype=mx.int32),
        mx.array([0, 0, 0, 0], dtype=mx.int32),
        n_in_rows=3,
        n_out_rows=2,
        n_kernels=1,
    )
    feats = mx.array(
        [[1.0, 2.0], [3.0, -4.0], [5.0, 6.0]],
        dtype=mx.float32,
    )

    assert pool_sum_edges(feats, mapping).tolist() == [
        [4.0, -2.0],
        [6.0, 8.0],
    ]
    assert pool_max_edges(feats, mapping).tolist() == [
        [3.0, 2.0],
        [5.0, 6.0],
    ]


def test_metal_exec_primitives_match_cpu_contract_when_available() -> None:
    def run() -> tuple[list[list[float]], list[list[float]]]:
        mapping = KernelMap(
            mx.array([0, 1, 2, 0], dtype=mx.int32),
            mx.array([0, 0, 1, 1], dtype=mx.int32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            n_in_rows=3,
            n_out_rows=2,
            n_kernels=2,
        )
        feats = mx.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=mx.float32,
        )
        weights = mx.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [1.0, 2.0]],
            ],
            dtype=mx.float32,
        )
        spmm = spmm_edges(feats, weights, mapping)
        pooled = pool_sum_edges(feats, mapping)
        mx.eval(spmm, pooled)
        return (
            cast('list[list[float]]', spmm.tolist()),
            cast('list[list[float]]', pooled.tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[11.0, 13.0], [9.0, 11.0]],
        [[4.0, 6.0], [6.0, 8.0]],
    )

from __future__ import annotations

from typing import cast

from mlx_lattice.core import CoordinateManager, SparseTensor
from mlx_lattice.ops import (
    build_generative_relation,
    build_kernel_relation,
    build_transposed_kernel_relation,
    downsample_coords,
    intersection_coords,
    kernel_offsets,
    kernel_relation,
    lookup_coords,
    union_coords,
)
from tests.support import mx, run_with_gpu_default


def test_coordinate_set_primitives_preserve_first_seen_order() -> None:
    lhs = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )
    rhs = mx.array(
        [[0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )

    assert downsample_coords(rhs, stride=2).tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    assert union_coords(lhs, rhs).tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
    ]
    assert intersection_coords(lhs, rhs).tolist() == [[0, 1, 0, 0]]
    assert lookup_coords(lhs, rhs).tolist() == [1, -1]


def test_kernel_offsets_and_relation_builders_emit_expected_edges() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))

    assert kernel_offsets((3, 1, 1)) == ((-1, 0, 0), (0, 0, 0), (1, 0, 0))
    assert relation.out_coords is not None
    assert relation.out_coords.tolist() == coords.tolist()
    assert relation.edge_coo.in_rows.tolist() == [0, 1, 0, 1, 2, 1, 2]
    assert relation.edge_coo.out_rows.tolist() == [1, 2, 0, 1, 2, 0, 1]
    assert relation.edge_coo.kernel_ids.tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_strided_and_transposed_relations_define_output_policy() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )
    strided = build_kernel_relation(coords, kernel_size=1, stride=2)
    generated = build_generative_relation(
        mx.array([[0, 1, 0, 0]], dtype=mx.int32),
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )
    transposed = build_transposed_kernel_relation(
        mx.array([[0, 1, 0, 0]], dtype=mx.int32),
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )

    assert strided.out_coords is not None
    assert strided.out_coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert strided.edge_coo.in_rows.tolist() == [0, 2]
    assert generated.out_coords is not None
    assert generated.out_coords.tolist() == [[0, 2, 0, 0], [0, 3, 0, 0]]
    assert transposed.out_coords is not None
    assert transposed.out_coords.tolist() == generated.out_coords.tolist()


def test_coordinate_manager_caches_tensor_kernel_relations() -> None:
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    manager = CoordinateManager()
    key = manager.insert_coords(coords)

    first = manager.kernel_relation(key, kernel_size=(3, 1, 1))
    second = manager.kernel_relation(key, kernel_size=(3, 1, 1))
    tensor_relation = kernel_relation(
        SparseTensor(coords, mx.ones((2, 1))), kernel_size=(3, 1, 1)
    )

    assert first is second
    assert tensor_relation.kernel_offsets == first.kernel_offsets


def test_metal_coordinate_primitives_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[list[list[int]], list[int], list[int], list[int]]:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        )
        relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))
        mx.eval(
            relation.out_coords,
            relation.edge_coo.in_rows,
            relation.edge_coo.out_rows,
            relation.edge_coo.kernel_ids,
        )
        assert relation.out_coords is not None
        return (
            cast('list[list[int]]', relation.out_coords.tolist()),
            cast('list[int]', relation.edge_coo.in_rows.tolist()),
            cast('list[int]', relation.edge_coo.out_rows.tolist()),
            cast('list[int]', relation.edge_coo.kernel_ids.tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        [0, 1, 0, 1, 2, 1, 2],
        [1, 2, 0, 1, 2, 0, 1],
        [0, 0, 1, 1, 1, 2, 2],
    )

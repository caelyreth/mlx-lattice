from __future__ import annotations

from typing import cast

from mlx_lattice.core import (
    CoordinateSet,
    OccupancyExpansion,
    SparseOccupancy,
)
from mlx_lattice.ops import (
    build_kernel_relation,
    build_knn_relation,
    build_radius_relation,
    build_target_kernel_relation,
    downsample_coords,
    intersection_coords,
    lookup_coords,
    union_coords,
)
from tests.cases.types import ValueCase
from tests.support import mx


def cases() -> list[ValueCase]:
    return [
        ValueCase('coords_kernel_relation', _kernel_relation),
        ValueCase('coords_target_relation', _target_relation),
        ValueCase(
            'coords_sets_duplicate_semantics', _sets_duplicate_semantics
        ),
        ValueCase('coords_strided_relation', _strided_relation),
        ValueCase('coords_neighbor_relations', _neighbor_relations),
    ]


def _active_rows(values: mx.array, count: mx.array) -> list[int]:
    return cast('list[int]', values[: int(count.tolist()[0])].tolist())


def _active_row_offsets(values: mx.array, count: mx.array) -> list[int]:
    return cast('list[int]', values[: int(count.tolist()[0]) + 1].tolist())


def _active_floats(values: mx.array, count: mx.array) -> list[float]:
    return cast('list[float]', values[: int(count.tolist()[0])].tolist())


def _active_coords(values: mx.array, count: mx.array) -> list[list[int]]:
    return cast(
        'list[list[int]]', values[: int(count.tolist()[0])].tolist()
    )


def _coord_set_rows(
    value: CoordinateSet | SparseOccupancy | OccupancyExpansion,
) -> list[list[int]]:
    return _active_coords(value.coords, value.active_rows)


def _relation_value(relation) -> object:
    mx.eval(
        relation.out_coords,
        relation.edges.in_rows,
        relation.edges.out_rows,
        relation.edges.kernel_ids,
        relation.row_offsets,
        relation.counts,
    )
    assert relation.out_coords is not None
    return (
        _active_coords(relation.out_coords, relation.out_count),
        _active_rows(relation.edges.in_rows, relation.edge_count),
        _active_rows(relation.edges.out_rows, relation.edge_count),
        _active_rows(relation.edges.kernel_ids, relation.edge_count),
        _active_row_offsets(relation.row_offsets, relation.out_count),
        cast('list[int]', relation.counts.tolist()),
    )


def _kernel_relation() -> object:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    return _relation_value(
        build_kernel_relation(coords, kernel_size=(3, 1, 1))
    )


def _target_relation() -> object:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    return _relation_value(
        build_target_kernel_relation(coords, target, kernel_size=(3, 1, 1))
    )


def _sets_duplicate_semantics() -> object:
    lhs = mx.array(
        [
            [0, 2, 0, 0],
            [0, -1, 0, 0],
            [0, 2, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=mx.int32,
    )
    rhs = mx.array(
        [
            [0, -1, 0, 0],
            [0, 3, 0, 0],
            [0, 3, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=mx.int32,
    )
    queries = mx.array(
        [[0, 2, 0, 0], [0, -1, 0, 0], [0, 9, 0, 0]],
        dtype=mx.int32,
    )
    empty = mx.array([], dtype=mx.int32).reshape((0, 4))
    downsampled = downsample_coords(lhs, stride=(2, 1, 1))
    union = union_coords(lhs, rhs)
    intersection = intersection_coords(lhs, rhs)
    lookup = lookup_coords(lhs, queries)
    empty_union = union_coords(empty, empty)
    mx.eval(
        downsampled.coords,
        downsampled.active_rows,
        union.coords,
        union.active_rows,
        intersection.coords,
        intersection.active_rows,
        lookup,
        empty_union.coords,
        empty_union.active_rows,
    )
    return (
        _coord_set_rows(downsampled),
        _coord_set_rows(union),
        _coord_set_rows(intersection),
        cast('list[int]', lookup.tolist()),
        _coord_set_rows(empty_union),
    )


def _strided_relation() -> object:
    coords = mx.array(
        [[0, row, 0, 0] for row in range(6)],
        dtype=mx.int32,
    )
    relation = build_kernel_relation(
        coords,
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
    )
    return _relation_value(relation)


def _neighbor_relations() -> object:
    source = mx.array(
        [
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 5, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=mx.int32,
    )
    query = mx.array(
        [[0, 1, 0, 0], [0, 4, 0, 0], [1, 1, 0, 0]],
        dtype=mx.int32,
    )
    knn = build_knn_relation(source, query, k=2)
    radius = build_radius_relation(source, query, radius=1.5)
    mx.eval(
        knn.edges.query_rows,
        knn.edges.source_rows,
        knn.edges.neighbor_ids,
        knn.distances,
        knn.row_offsets,
        knn.counts,
        radius.counts,
    )
    return (
        _active_rows(knn.edges.query_rows, knn.edge_count),
        _active_rows(knn.edges.source_rows, knn.edge_count),
        _active_rows(knn.edges.neighbor_ids, knn.edge_count),
        _active_floats(knn.distances, knn.edge_count),
        cast('list[int]', knn.counts.tolist()),
        cast('list[int]', radius.counts.tolist()),
    )

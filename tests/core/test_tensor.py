from __future__ import annotations

from typing import Any, cast

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.core import CoordinateManager
from mlx_lattice.core import SparseTensor as CoreSparseTensor
from mlx_lattice.ops import (
    align_sparse,
    cat,
    contains_coords,
    crop,
    lookup_coords,
    prune,
    prune_mask,
    reindex_sparse,
    sparse_add,
    sparse_cat_aligned,
    sparse_collate,
    sparse_from_coordinates,
    sparse_mul,
    sparse_sub,
    topk_rows,
)
from tests.support import mx


def _active_count(value: mx.array) -> int:
    return int(value.item())


def test_sparse_tensor_owns_identity_and_validates_shape() -> None:
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    feats = mx.ones((1, 2), dtype=mx.float32)

    x = SparseTensor(coords, feats, stride=(1, 2, 3))

    assert SparseTensor is CoreSparseTensor
    assert x.coords is coords
    assert x.feats is feats
    assert x.stride == (1, 2, 3)
    assert x.shape == (1, 2)
    assert x.dtype == mx.float32

    with pytest.raises(ValueError, match='same row count'):
        SparseTensor(coords, mx.ones((2, 2), dtype=mx.float32))


def test_sparse_tensor_reuses_and_rejects_coordinate_ownership() -> None:
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.ones((2, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats)
    y = x.replace(feats=feats + 1)

    reused = y.reuse_coords_from(x)

    assert reused.coord_key == x.coord_key
    assert reused.coord_manager is x.coord_manager
    assert reused.coords is x.coords

    equal_values = SparseTensor(
        mx.array(coords.tolist(), dtype=mx.int32), feats + 2
    )
    same_array = SparseTensor(coords, feats + 3)
    assert not x.same_coords(equal_values)
    assert not x.same_coords(same_array)
    with pytest.raises(ValueError, match='coordinates must match'):
        equal_values.reuse_coords_from(x)

    manager = CoordinateManager()
    key = manager.insert_coords(coords)
    with pytest.raises(ValueError, match='coord_manager is required'):
        SparseTensor(coords, feats, coord_key=key)
    with pytest.raises(ValueError, match='coord_key'):
        SparseTensor(
            coords, feats, coord_key=key, coord_manager=CoordinateManager()
        )
    with pytest.raises(ValueError, match='manager-owned array'):
        SparseTensor(
            mx.array(coords.tolist(), dtype=mx.int32),
            feats,
            coord_key=key,
            coord_manager=manager,
        )


def test_sparse_tensor_coordinate_queries_and_feature_replacement() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        dtype=mx.int32,
    )
    x = SparseTensor(coords, mx.ones((3, 1), dtype=mx.float32))
    queries = mx.array(
        [[0, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0]],
        dtype=mx.int32,
    )

    out = x.replace(feats=x.feats + 1).astype(mx.float16)

    assert lookup_coords(x.coords, queries).tolist() == [1, -1, 2]
    assert contains_coords(x.coords, queries).tolist() == [
        True,
        False,
        True,
    ]
    assert out.dtype == mx.float16
    assert out.coord_key == x.coord_key


def test_tensor_ops_preserve_or_create_identity_intentionally() -> None:
    coords = mx.array(
        [[0, 3, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[3.0], [1.0], [2.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    y = x.replace(feats=feats * 2)

    kept = prune(x, mx.array([2, 0], dtype=mx.int32))
    joined = cat([x, y])

    assert kept.coords.tolist() == [[0, 2, 0, 0], [0, 3, 0, 0]]
    assert kept.coord_manager is x.coord_manager
    assert kept.coord_key != x.coord_key
    assert joined.feats.tolist() == [[3.0, 6.0], [1.0, 2.0], [2.0, 4.0]]
    assert joined.coord_key == x.coord_key


def test_sparse_alignment_joins_by_coordinate_value() -> None:
    lhs = SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[10.0], [20.0], [30.0]], dtype=mx.float32),
    )
    rhs = SparseTensor(
        mx.array(
            [[0, 2, 0, 0], [0, 3, 0, 0], [0, 1, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[300.0], [400.0], [200.0]], dtype=mx.float32),
    )

    inner = align_sparse(lhs, rhs, join='inner')
    outer = align_sparse(lhs, rhs, join='outer')

    assert inner.coords[: _active_count(inner.active_rows)].tolist() == [
        [0, 1, 0, 0],
        [0, 2, 0, 0],
    ]
    assert inner.lhs_rows[:2].tolist() == [1, 2]
    assert inner.rhs_rows[:2].tolist() == [2, 0]
    assert outer.coords[: _active_count(outer.active_rows)].tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 3, 0, 0],
    ]
    assert outer.lhs_rows[:4].tolist() == [0, 1, 2, -1]
    assert outer.rhs_rows[:4].tolist() == [-1, 2, 0, 1]


def test_sparse_add_uses_value_aligned_outer_union() -> None:
    lhs = SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[1.0], [2.0]], dtype=mx.float32),
    )
    rhs = SparseTensor(
        mx.array([[0, 1, 0, 0], [0, 2, 0, 0]], dtype=mx.int32),
        mx.array([[20.0], [30.0]], dtype=mx.float32),
    )

    out = lhs + rhs
    explicit = sparse_add(lhs, rhs)
    difference = lhs - rhs
    explicit_difference = sparse_sub(lhs, rhs)

    assert out.coords[: _active_count(out.active_rows)].tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
    ]
    assert out.feats[:3].tolist() == [[1.0], [22.0], [30.0]]
    assert explicit.feats[:3].tolist() == out.feats[:3].tolist()
    assert difference.feats[:3].tolist() == [[1.0], [-18.0], [-30.0]]
    assert explicit_difference.feats[:3].tolist() == (
        difference.feats[:3].tolist()
    )


def test_sparse_cat_and_mul_can_value_align() -> None:
    lhs = SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[1.0], [2.0]], dtype=mx.float32),
    )
    rhs = SparseTensor(
        mx.array([[0, 1, 0, 0], [0, 2, 0, 0]], dtype=mx.int32),
        mx.array([[20.0], [30.0]], dtype=mx.float32),
    )

    joined = cat([lhs, rhs], join='outer')
    inner = sparse_cat_aligned(lhs, rhs)
    multiplied = sparse_mul(lhs, rhs)
    operator_multiplied = lhs * rhs
    intersection_multiplied = lhs & rhs

    assert joined.coords[:3].tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
    ]
    assert joined.feats[:3].tolist() == [
        [1.0, 0.0],
        [2.0, 20.0],
        [0.0, 30.0],
    ]
    assert inner.feats[:1].tolist() == [[2.0, 20.0]]
    assert multiplied.feats[:1].tolist() == [[40.0]]
    assert operator_multiplied.feats[:1].tolist() == [[40.0]]
    assert intersection_multiplied.feats[:1].tolist() == [[40.0]]


def test_value_alignment_respects_active_rows() -> None:
    lhs = SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 9, 0, 0]], dtype=mx.int32),
        mx.array([[1.0], [9.0]], dtype=mx.float32),
        active_rows=mx.array([1], dtype=mx.int32),
    )
    rhs = SparseTensor(
        mx.array([[0, 9, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[90.0], [2.0]], dtype=mx.float32),
        active_rows=mx.array([1], dtype=mx.int32),
    )

    out = sparse_add(lhs, rhs)

    assert out.coords[: _active_count(out.active_rows)].tolist() == [
        [0, 0, 0, 0],
        [0, 9, 0, 0],
    ]
    assert out.feats[:2].tolist() == [[1.0], [90.0]]


def test_crop_filters_by_spatial_bounds_and_active_rows() -> None:
    x = SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[0.0], [2.0], [4.0], [99.0]], dtype=mx.float32),
        active_rows=mx.array([3], dtype=mx.int32),
    )

    out = crop(x, min_coord=(1, 0, 0), max_coord=(3, 0, 0))

    assert out.coords.tolist() == [[0, 2, 0, 0]]
    assert out.feats.tolist() == [[2.0]]


def test_prune_mask_selects_sparse_rows_by_boolean_mask() -> None:
    coords = mx.array(
        [[0, 3, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[3.0], [1.0], [2.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)

    kept = prune_mask(x, mx.array([True, False, True], dtype=mx.bool_))

    assert kept.coords.tolist() == [[0, 3, 0, 0], [0, 2, 0, 0]]
    assert kept.feats.tolist() == [[3.0], [2.0]]


def test_reindex_sparse_preserves_target_order_and_fills_missing_rows() -> (
    None
):
    source = SparseTensor(
        mx.array(
            [[0, 1, 0, 0], [0, 3, 0, 0], [0, 5, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [3.0], [5.0]], dtype=mx.float32),
    )
    target = SparseTensor(
        mx.array(
            [[0, 5, 0, 0], [0, 2, 0, 0], [0, 1, 0, 0]],
            dtype=mx.int32,
        ),
        mx.zeros((3, 4), dtype=mx.float32),
        batch_counts=(3,),
    )

    out = reindex_sparse(source, target, fill=-2.0)

    assert out.same_coords(target)
    assert out.batch_counts == (3,)
    assert out.feats.tolist() == [[5.0], [-2.0], [1.0]]


def test_sparse_collate_decompose_topk_and_prune() -> None:
    x = sparse_collate(
        [
            mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32),
            mx.array([[2, 0, 0], [3, 0, 0]], dtype=mx.int32),
        ],
        [
            mx.array([[0.5], [2.0]], dtype=mx.float32),
            mx.array([[3.0], [1.0]], dtype=mx.float32),
        ],
    )

    out = prune(x, topk_rows(x, [1, 1]))

    assert x.coords.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 2, 0, 0],
        [1, 3, 0, 0],
    ]
    assert [part.tolist() for part in x.decomposed_coordinates] == [
        [[0, 0, 0], [1, 0, 0]],
        [[2, 0, 0], [3, 0, 0]],
    ]
    assert out.feats.tolist() == [[2.0], [3.0]]


def test_sparse_construction_averages_duplicate_coordinates() -> None:
    coords = mx.array(
        [[0, 2, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )

    def loss(feats: mx.array) -> mx.array:
        return mx.sum(
            sparse_from_coordinates(
                coords,
                feats,
                batch_counts=(3,),
                duplicate_reduction='mean',
            ).feats
        )

    feats = mx.array([[2.0], [5.0], [6.0]], dtype=mx.float32)
    out = sparse_from_coordinates(
        coords,
        feats,
        batch_counts=(3,),
        duplicate_reduction='mean',
    )
    gradient = mx.grad(loss)(feats)

    assert out.coords.tolist() == [[0, 2, 0, 0], [0, 1, 0, 0]]
    assert out.feats.tolist() == [[4.0], [5.0]]
    assert out.batch_counts == (2,)
    assert gradient.tolist() == [[0.5], [1.0], [0.5]]


def test_sparse_collate_applies_duplicate_reduction_per_batch() -> None:
    out = sparse_collate(
        [
            mx.array([[0, 0, 0], [0, 0, 0]], dtype=mx.int32),
            mx.array([[0, 0, 0]], dtype=mx.int32),
        ],
        [
            mx.array([[2.0], [6.0]], dtype=mx.float32),
            mx.array([[9.0]], dtype=mx.float32),
        ],
        duplicate_reduction='mean',
    )

    assert out.coords.tolist() == [[0, 0, 0, 0], [1, 0, 0, 0]]
    assert out.feats.tolist() == [[4.0], [9.0]]
    assert out.batch_counts == (1, 1)


def test_sparse_construction_validates_duplicate_reduction() -> None:
    with pytest.raises(ValueError, match='duplicate_reduction'):
        sparse_from_coordinates(
            mx.zeros((1, 4), dtype=mx.int32),
            mx.zeros((1, 1), dtype=mx.float32),
            duplicate_reduction=cast(Any, 'sum'),
        )


def test_batch_partitioned_views_infer_noncontiguous_rows_from_coords() -> (
    None
):
    coords = mx.array(
        [[1, 1, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0]],
        dtype=mx.int32,
    )
    x = SparseTensor(
        coords,
        mx.array([[10.0], [20.0], [30.0]], dtype=mx.float32),
    )

    decomposed_coords, decomposed_feats = (
        x.decomposed_coordinates_and_features
    )

    assert [part.tolist() for part in x.batch_rows] == [[1], [0, 2]]
    assert [part.tolist() for part in decomposed_coords] == [
        [[0, 0, 0]],
        [[1, 0, 0], [2, 0, 0]],
    ]
    assert [part.tolist() for part in decomposed_feats] == [
        [[20.0]],
        [[10.0], [30.0]],
    ]


def test_prune_preserves_row_order_and_declared_empty_batches() -> None:
    x = SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
        batch_counts=(1, 2, 0),
    )

    out = prune(x, mx.array([2, 0], dtype=mx.int32))

    assert out.coords.tolist() == [[1, 2, 0, 0], [0, 0, 0, 0]]
    assert out.feats.tolist() == [[3.0], [1.0]]
    assert out.batch_counts == (1, 1, 0)
    assert [part.tolist() for part in out.batch_rows] == [[1], [0], []]


def test_prune_mask_rejects_inactive_selection() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 9, 0, 0]], dtype=mx.int32),
        mx.ones((2, 1), dtype=mx.float32),
        active_rows=mx.array([1], dtype=mx.int32),
    )

    with pytest.raises(ValueError, match='inactive'):
        prune_mask(x, mx.array([True, True], dtype=mx.bool_))

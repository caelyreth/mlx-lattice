from __future__ import annotations

from typing import Any, cast

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    build_point_voxel_map,
    devoxelize,
    interpolate_point_features,
    sparse_quantize,
    voxelize,
    voxelize_with_quantization,
)
from tests.support import (
    active_coords,
    active_feats,
    assert_nested_close,
    mx,
)

pytestmark = [
    pytest.mark.ops,
    pytest.mark.quantization,
    pytest.mark.usefixtures('selected_backend'),
]


def _active_rows(values: mx.array, count: mx.array) -> list[int]:
    return cast('list[int]', values[: int(count.tolist()[0])].tolist())


def _active_coords(values: mx.array, count: mx.array) -> list[list[int]]:
    return cast(
        'list[list[int]]', values[: int(count.tolist()[0])].tolist()
    )


def test_sparse_quantize_defines_ordered_voxel_metadata_contract() -> None:
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
    batches = mx.array([0, 0, 0, 0, 1], dtype=mx.int32)

    quantized = sparse_quantize(
        points,
        voxel_size=(1.0, 1.0, 1.0),
        batch_indices=batches,
    )

    assert quantized.capacity == 5
    assert quantized.active_count is quantized.active_rows
    assert quantized.active_rows.tolist() == [4]
    assert _active_coords(quantized.coords, quantized.active_count) == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0],
        [1, 0, 2, 0],
    ]
    assert quantized.inverse_rows.tolist() == [0, 0, 1, 2, 3]
    assert _active_rows(quantized.counts, quantized.active_count) == [
        2,
        1,
        1,
        1,
    ]


def test_sparse_quantize_respects_origin_and_active_rows() -> None:
    points = mx.array(
        [
            [0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [99.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    active_rows = mx.array([3], dtype=mx.int32)

    quantized = sparse_quantize(
        points,
        voxel_size=(2.0, 1.0, 1.0),
        origin=(0.5, 0.0, 0.0),
        active_rows=active_rows,
    )

    assert quantized.active_rows.tolist() == [2]
    assert _active_coords(quantized.coords, quantized.active_count) == [
        [0, -1, 0, 0],
        [0, 0, 0, 0],
    ]
    assert quantized.inverse_rows.tolist() == [0, 1, 1, -1]
    assert quantized.counts.tolist() == [1, 2, 0, 0]


def test_voxelize_aggregates_features_into_sparse_tensor() -> None:
    points = mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )
    feats = mx.array(
        [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]],
        dtype=mx.float32,
    )

    mean = voxelize(points, feats, voxel_size=1.0, reduction='mean')
    summed = voxelize(points, feats, voxel_size=1.0, reduction='sum')

    assert active_coords(mean) == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert_nested_close(
        active_feats(mean).tolist(), [[2.0, 20.0], [5.0, 50.0]]
    )
    assert_nested_close(
        active_feats(summed).tolist(), [[4.0, 40.0], [5.0, 50.0]]
    )
    assert mean.stride == (1, 1, 1)


def test_voxelize_with_quantization_reuses_native_map() -> None:
    points = mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )
    feats = mx.array(
        [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]],
        dtype=mx.float32,
    )
    quantization = sparse_quantize(points)
    template = SparseTensor(
        quantization.coords,
        mx.zeros_like(feats),
        active_rows=quantization.active_rows,
    )

    mapped = voxelize_with_quantization(
        quantization,
        feats,
        reduction='mean',
        template=template,
    )
    direct = voxelize(points, feats, reduction='mean')

    assert mapped.same_coords(template)
    assert active_coords(mapped) == active_coords(direct)
    assert_nested_close(
        active_feats(mapped).tolist(), active_feats(direct).tolist()
    )


def _voxelize_feature_case():
    points = mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )
    feats = mx.array([[1.0], [3.0], [5.0]], dtype=mx.float32)
    tangent = mx.array([[10.0], [20.0], [30.0]], dtype=mx.float32)

    def features(feats_arg: mx.array) -> mx.array:
        return voxelize(points, feats_arg, reduction='mean').feats

    def loss(feats_arg: mx.array) -> mx.array:
        return mx.sum(features(feats_arg))

    return feats, tangent, features, loss


def test_voxelize_feature_aggregation_is_autogradable() -> None:
    feats, tangent, features, loss = _voxelize_feature_case()
    grads = mx.grad(loss)(feats)
    _, jvps = mx.jvp(features, [feats], [tangent])
    jvp_values = jvps[0] if isinstance(jvps, list) else jvps

    assert_nested_close(grads.tolist(), [[0.5], [0.5], [1.0]])
    assert_nested_close(jvp_values.tolist()[:2], [[15.0], [30.0]])


def test_voxelize_feature_aggregation_supports_mx_compile(
    compile_backend,
) -> None:
    feats, _, features, _ = _voxelize_feature_case()

    compiled = mx.compile(features)(feats)
    assert_nested_close(compiled.tolist()[:2], [[2.0], [5.0]])


def test_devoxelize_nearest_samples_sparse_voxel_features() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array(
        [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]],
        dtype=mx.float32,
    )
    voxels = SparseTensor(coords, feats)
    points = mx.array(
        [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0], [9.0, 0.0, 0.0]],
        dtype=mx.float32,
    )

    sampled = devoxelize(points, voxels, interpolation='nearest')

    assert_nested_close(
        sampled.tolist(),
        [[10.0, 1.0], [20.0, 2.0], [0.0, 0.0]],
    )


def test_devoxelize_linear_interpolates_voxel_corners() -> None:
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=mx.int32,
    )
    feats = mx.array([[float(row)] for row in range(8)], dtype=mx.float32)
    voxels = SparseTensor(coords, feats)
    points = mx.array([[0.5, 0.5, 0.5]], dtype=mx.float32)

    sampled = devoxelize(points, voxels, interpolation='linear')

    assert_nested_close(sampled.tolist(), [[3.5]])


def test_point_voxel_map_reuse_and_active_rows() -> None:
    coords = mx.array(
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[2.0], [4.0], [99.0]], dtype=mx.float32)
    active_rows = mx.array([2], dtype=mx.int32)
    voxels = SparseTensor(coords, feats, active_rows=active_rows)
    points = mx.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=mx.float32)
    point_rows = mx.array([1], dtype=mx.int32)

    point_map = build_point_voxel_map(
        points,
        voxels.coords,
        voxels.active_rows,
        point_active_rows=point_rows,
        interpolation='nearest',
    )
    sampled = interpolate_point_features(voxels.feats, point_map)

    assert point_map.point_count == 2
    assert_nested_close(sampled.tolist(), [[4.0], [0.0]])


def test_devoxelize_feature_interpolation_is_autogradable() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )
    points = mx.array([[0.25, 0.0, 0.0]], dtype=mx.float32)
    point_map = build_point_voxel_map(
        points,
        coords,
        mx.array([2], dtype=mx.int32),
        interpolation='linear',
    )
    feats = mx.array([[2.0], [6.0]], dtype=mx.float32)
    tangent = mx.array([[10.0], [20.0]], dtype=mx.float32)

    def features(feats_arg: mx.array) -> mx.array:
        return interpolate_point_features(feats_arg, point_map)

    def loss(feats_arg: mx.array) -> mx.array:
        return mx.sum(features(feats_arg))

    grads = mx.grad(loss)(feats)
    _, jvps = mx.jvp(features, [feats], [tangent])
    jvp_values = jvps[0] if isinstance(jvps, list) else jvps

    assert_nested_close(features(feats).tolist(), [[3.0]])
    assert_nested_close(grads.tolist(), [[0.75], [0.25]])
    assert_nested_close(jvp_values.tolist(), [[12.5]])


def test_quantization_ops_reject_ambiguous_contracts() -> None:
    points = mx.ones((2, 3), dtype=mx.float32)
    feats = mx.ones((2, 1), dtype=mx.float32)

    with pytest.raises(ValueError, match='points must have shape'):
        sparse_quantize(mx.ones((2, 4), dtype=mx.float32))
    with pytest.raises(ValueError, match='points must be float32'):
        sparse_quantize(mx.ones((2, 3), dtype=mx.int32))
    with pytest.raises(ValueError, match='batch_indices'):
        sparse_quantize(points, batch_indices=mx.ones((2,), dtype=mx.int64))
    with pytest.raises(ValueError, match='voxel_size'):
        sparse_quantize(points, voxel_size=0.0)
    with pytest.raises(ValueError, match='reduction'):
        voxelize(points, feats, reduction=cast('str', 'max'))
    with pytest.raises(ValueError, match='matching rows'):
        voxelize(points, mx.ones((3, 1), dtype=mx.float32))
    with pytest.raises(ValueError, match='interpolation'):
        devoxelize(
            points,
            SparseTensor(
                mx.array([[0, 0, 0, 0]], dtype=mx.int32),
                mx.ones((1, 1), dtype=mx.float32),
            ),
            interpolation=cast('Any', 'cubic'),
        )

from __future__ import annotations

from typing import Any

import mlx.nn as mxnn

from mlx_lattice import SparseTensor
from mlx_lattice import nn as lnn
from mlx_lattice.ops import sparse_collate
from tests.cases.types import ValueCase
from tests.support import active_coords, active_feats, mx


def cases() -> list[ValueCase]:
    return [
        ValueCase('nn_feature_modules', _feature_modules),
        ValueCase('nn_conv_modules', _conv_modules),
        ValueCase('nn_conv_transforms', _conv_transforms),
        ValueCase('nn_pool_transforms', _pool_transforms),
        ValueCase('nn_global_pool_modules', _global_pool_modules),
    ]


def _tensor() -> SparseTensor:
    return SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
    )


def _feature_modules() -> object:
    x = _tensor()
    layer = lnn.Linear(2, 2)
    layer.weight = mx.array([[2.0, 3.0], [5.0, 7.0]], dtype=mx.float32)
    layer.bias = mx.array([1.0, -1.0], dtype=mx.float32)
    relu = lnn.ReLU()

    out = relu(layer(x))
    mx.eval(out.feats)
    return out.feats.tolist()


def _conv_modules() -> object:
    x = SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
    )
    conv = lnn.Conv3d(1, 1, kernel_size=(3, 1, 1), bias=False)
    conv.weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)
    subm = lnn.SubmConv3d(1, 1, kernel_size=(3, 1, 1), bias=False)
    subm.weight = conv.weight
    out = conv(x)
    subm_out = subm(x)
    mx.eval(out.coords, out.feats, out.active_rows, subm_out.feats)
    return (
        active_coords(out),
        active_feats(out).tolist(),
        subm_out.feats.tolist(),
    )


def _conv_transforms() -> object:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    conv = lnn.Conv3d(1, 1, kernel_size=(3, 1, 1), bias=False)
    conv.weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    def features(feats_arg: mx.array) -> mx.array:
        return conv(SparseTensor(coords, feats_arg)).feats

    def loss(model: Any, feats_arg: mx.array) -> mx.array:
        return mx.sum(model(SparseTensor(coords, feats_arg)).feats)

    value, param_grads = mxnn.value_and_grad(conv, loss)(conv, feats)
    outputs, input_vjps = mx.vjp(
        features,
        [feats],
        [mx.ones((3, 1), dtype=mx.float32)],
    )
    _, input_jvps = mx.jvp(features, [feats], [mx.ones_like(feats)])
    mx.eval(
        value,
        param_grads['weight'],
        outputs[0],
        input_vjps[0],
        input_jvps[0],
    )
    return (
        outputs[0].tolist(),
        value.tolist(),
        input_vjps[0].tolist(),
        input_jvps[0].tolist(),
        param_grads['weight'].tolist(),
    )


def _pool_transforms() -> object:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[2.0], [2.0], [1.0]], dtype=mx.float32)
    tangent = mx.array([[10.0], [20.0], [30.0]], dtype=mx.float32)
    max_pool = lnn.MaxPool3d(kernel_size=(3, 1, 1), stride=1)

    def maxed(feats_arg: mx.array) -> mx.array:
        return max_pool(SparseTensor(coords, feats_arg)).feats

    outputs, input_vjps = mx.vjp(
        maxed,
        [feats],
        [mx.ones((3, 1), dtype=mx.float32)],
    )
    _, input_jvps = mx.jvp(maxed, [feats], [tangent])
    mx.eval(outputs[0], input_vjps[0], input_jvps[0])
    return (
        outputs[0].tolist(),
        input_vjps[0].tolist(),
        input_jvps[0].tolist(),
    )


def _global_pool_modules() -> object:
    x = sparse_collate(
        [
            mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32),
            mx.array([[2, 0, 0], [3, 0, 0]], dtype=mx.int32),
        ],
        [
            mx.array([[1.0], [2.0]], dtype=mx.float32),
            mx.array([[3.0], [5.0]], dtype=mx.float32),
        ],
    )
    summed = lnn.GlobalSumPool()(x)
    averaged = lnn.GlobalAvgPool()(x)
    maxed = lnn.GlobalMaxPool()(x)
    mx.eval(summed, averaged, maxed)
    return summed.tolist(), averaged.tolist(), maxed.tolist()

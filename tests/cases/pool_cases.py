from __future__ import annotations

from typing import cast

from mlx_lattice import SparseTensor
from mlx_lattice.ops import avg_pool3d, max_pool3d, sum_pool3d
from tests.cases.types import ValueCase
from tests.support import active_coords, active_feats, mx


def cases() -> list[ValueCase]:
    return [
        ValueCase('pool_local_modes', _local_modes),
        ValueCase('pool_gradients', _gradients),
        ValueCase('pool_max_tie_policy', _max_tie_policy),
        ValueCase('pool_jvp', _jvp),
        ValueCase(
            'pool_strided_autodiff_topology', _strided_autodiff_topology
        ),
        ValueCase('pool_active_rows', _active_rows),
    ]


def _line_inputs() -> tuple[mx.array, mx.array]:
    return (
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
    )


def _local_modes() -> object:
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
    mx.eval(summed.feats, maxed.feats, averaged.feats)
    return (
        active_feats(summed).tolist(),
        active_feats(maxed).tolist(),
        active_feats(averaged).tolist(),
    )


def _gradients() -> object:
    coords, feats = _line_inputs()

    def sum_loss(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(sum_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats)

    def avg_loss(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(avg_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats)

    def max_loss(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(max_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats)

    sum_grad = mx.grad(sum_loss)(feats)
    avg_grad = mx.grad(avg_loss)(feats)
    max_grad = mx.grad(max_loss)(feats)
    mx.eval(sum_grad, avg_grad, max_grad)
    return sum_grad.tolist(), avg_grad.tolist(), max_grad.tolist()


def _max_tie_policy() -> object:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[2.0], [2.0], [1.0]], dtype=mx.float32)
    tangent = mx.array([[10.0], [20.0], [30.0]], dtype=mx.float32)

    def pooled(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return max_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

    grad = mx.grad(lambda value: mx.sum(pooled(value)))(feats)
    _, jvps = mx.jvp(pooled, [feats], [tangent])
    mx.eval(grad, jvps[0])
    return grad.tolist(), jvps[0].tolist()


def _jvp() -> object:
    coords, feats = _line_inputs()
    tangent = mx.ones_like(feats)

    def summed(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return sum_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

    def averaged(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return avg_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

    def maxed(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return max_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

    _, sum_jvp = mx.jvp(summed, [feats], [tangent])
    _, avg_jvp = mx.jvp(averaged, [feats], [tangent])
    _, max_jvp = mx.jvp(maxed, [feats], [tangent])
    mx.eval(sum_jvp[0], avg_jvp[0], max_jvp[0])
    return sum_jvp[0].tolist(), avg_jvp[0].tolist(), max_jvp[0].tolist()


def _strided_autodiff_topology() -> object:
    coords = mx.array(
        [[0, row, 0, 0] for row in range(8)],
        dtype=mx.int32,
    )
    feats = mx.array(
        [[float(row)] for row in range(1, 9)], dtype=mx.float32
    )
    tangent = mx.array(
        [[float(row * 10)] for row in range(1, 9)],
        dtype=mx.float32,
    )

    def summed(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return sum_pool3d(x, kernel_size=2, stride=2).feats

    def averaged(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return avg_pool3d(x, kernel_size=2, stride=2).feats

    def maxed(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return max_pool3d(x, kernel_size=2, stride=2).feats

    sum_grad = mx.grad(lambda value: mx.sum(summed(value)))(feats)
    avg_grad = mx.grad(lambda value: mx.sum(averaged(value)))(feats)
    max_grad = mx.grad(lambda value: mx.sum(maxed(value)))(feats)
    _, avg_jvp = mx.jvp(averaged, [feats], [tangent])
    _, max_jvp = mx.jvp(maxed, [feats], [tangent])
    mx.eval(sum_grad, avg_grad, max_grad, avg_jvp[0], max_jvp[0])
    return (
        sum_grad.tolist(),
        avg_grad.tolist(),
        max_grad.tolist(),
        avg_jvp[0][:4].tolist(),
        max_jvp[0][:4].tolist(),
    )


def _active_rows() -> object:
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 99, 0, 0],
            [0, 100, 0, 0],
        ],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [100.0], [200.0]], dtype=mx.float32)
    x = SparseTensor(
        coords,
        feats,
        active_rows=mx.array([2], dtype=mx.int32),
    )
    out = sum_pool3d(x, kernel_size=(3, 1, 1), stride=1)
    mx.eval(out.coords, out.feats, out.active_rows)
    return (
        active_coords(out),
        cast('list[list[float]]', active_feats(out).tolist()),
        cast('list[int]', out.active_rows.tolist()),
    )

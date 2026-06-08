from __future__ import annotations

from typing import cast

from mlx_lattice.core import KernelRelation
from mlx_lattice.ops import (
    build_kernel_relation,
)
from mlx_lattice.ops._exec import (
    execute_pool_max,
    execute_pool_sum,
    execute_spmm,
)
from tests.support import assert_nested_close, mx, run_with_gpu_default


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


def test_execute_spmm_matches_manual_reference_with_repeated_outputs() -> (
    None
):
    relation = KernelRelation(
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

    out = execute_spmm(feats, weights, relation)

    assert out.tolist() == _manual_spmm(
        cast('list[list[float]]', feats.tolist()),
        cast('list[list[list[float]]]', weights.tolist()),
        cast('list[int]', relation.edge_coo.in_rows.tolist()),
        cast('list[int]', relation.edge_coo.out_rows.tolist()),
        cast('list[int]', relation.edge_coo.kernel_ids.tolist()),
        2,
    )


def test_execute_spmm_consumes_lazy_kernel_relation_outputs() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weights = mx.ones((3, 1, 1), dtype=mx.float32)

    assert execute_spmm(feats, weights, relation).tolist() == [
        [3.0],
        [6.0],
        [5.0],
    ]


def test_execute_spmm_gradients_match_manual_reference() -> None:
    relation = KernelRelation(
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
    cotangent = mx.array([[1.0, 3.0], [5.0, 7.0]], dtype=mx.float32)

    def loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        return mx.sum(
            execute_spmm(feats_arg, weight_arg, relation) * cotangent
        )

    grad_feats, grad_weights = mx.grad(loss, argnums=(0, 1))(feats, weights)

    assert grad_feats.tolist() == [[18.0, 22.0], [5.0, 7.0], [5.0, 7.0]]
    assert grad_weights.tolist() == [
        [[26.0, 38.0], [32.0, 48.0]],
        [[8.0, 16.0], [14.0, 26.0]],
    ]


def test_execute_spmm_jvp_matches_linearized_reference() -> None:
    relation = KernelRelation(
        mx.array([0], dtype=mx.int32),
        mx.array([0], dtype=mx.int32),
        mx.array([0], dtype=mx.int32),
        n_in_rows=1,
        n_out_rows=1,
        n_kernels=1,
    )
    feats = mx.array([[2.0]], dtype=mx.float32)
    weights = mx.array([[[3.0]]], dtype=mx.float32)
    feat_tangent = mx.array([[5.0]], dtype=mx.float32)
    weight_tangent = mx.array([[[7.0]]], dtype=mx.float32)

    primals, tangents = mx.jvp(
        lambda feats_arg, weight_arg: execute_spmm(
            feats_arg,
            weight_arg,
            relation,
        ),
        [feats, weights],
        [feat_tangent, weight_tangent],
    )

    assert primals[0].tolist() == [[6.0]]
    assert tangents[0].tolist() == [[29.0]]


def test_pool_edge_reductions_match_manual_reference() -> None:
    relation = KernelRelation(
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

    assert execute_pool_sum(feats, relation).tolist() == [
        [4.0, -2.0],
        [6.0, 8.0],
    ]
    assert execute_pool_max(feats, relation).tolist() == [
        [3.0, 2.0],
        [5.0, 6.0],
    ]


def test_pool_edge_reduction_gradients_match_contract() -> None:
    relation = KernelRelation(
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
    cotangent = mx.array([[10.0, 20.0], [30.0, 40.0]], dtype=mx.float32)

    def sum_loss(feats_arg: mx.array) -> mx.array:
        return mx.sum(execute_pool_sum(feats_arg, relation) * cotangent)

    def max_loss(feats_arg: mx.array) -> mx.array:
        return mx.sum(execute_pool_max(feats_arg, relation) * cotangent)

    assert mx.grad(sum_loss)(feats).tolist() == [
        [40.0, 60.0],
        [10.0, 20.0],
        [30.0, 40.0],
    ]
    assert mx.grad(max_loss)(feats).tolist() == [
        [0.0, 20.0],
        [10.0, 0.0],
        [30.0, 40.0],
    ]


def test_metal_exec_primitives_match_cpu_contract_when_available() -> None:
    def run() -> tuple[list[list[float]], list[list[float]]]:
        relation = KernelRelation(
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
        spmm = execute_spmm(feats, weights, relation)
        pooled = execute_pool_sum(feats, relation)
        mx.eval(spmm, pooled)
        return (
            cast('list[list[float]]', spmm.tolist()),
            cast('list[list[float]]', pooled.tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[11.0, 13.0], [9.0, 11.0]],
        [[4.0, 6.0], [6.0, 8.0]],
    )


def test_metal_exec_primitive_gradients_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[
        list[list[float]],
        list[list[list[float]]],
        list[list[float]],
    ]:
        relation = KernelRelation(
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
        cotangent = mx.array([[1.0, 3.0], [5.0, 7.0]], dtype=mx.float32)

        def spmm_loss(
            feats_arg: mx.array,
            weight_arg: mx.array,
        ) -> mx.array:
            return mx.sum(
                execute_spmm(feats_arg, weight_arg, relation) * cotangent
            )

        def pool_loss(feats_arg: mx.array) -> mx.array:
            return mx.sum(execute_pool_sum(feats_arg, relation) * cotangent)

        grad_feats, grad_weights = mx.grad(spmm_loss, argnums=(0, 1))(
            feats,
            weights,
        )
        pool_grad = mx.grad(pool_loss)(feats)
        mx.eval(grad_feats, grad_weights, pool_grad)
        return (
            cast('list[list[float]]', grad_feats.tolist()),
            cast('list[list[list[float]]]', grad_weights.tolist()),
            cast('list[list[float]]', pool_grad.tolist()),
        )

    grad_feats, grad_weights, pool_grad = run_with_gpu_default(run)

    assert_nested_close(
        grad_feats,
        [[18.0, 22.0], [5.0, 7.0], [5.0, 7.0]],
    )
    assert_nested_close(
        grad_weights,
        [[[26.0, 38.0], [32.0, 48.0]], [[8.0, 16.0], [14.0, 26.0]]],
    )
    assert_nested_close(pool_grad, [[6.0, 10.0], [1.0, 3.0], [5.0, 7.0]])

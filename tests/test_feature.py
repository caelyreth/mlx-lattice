from __future__ import annotations

# ruff: noqa: E402, I001

from typing import Any, cast

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    batch_norm,
    dropout,
    gelu,
    layer_norm,
    leaky_relu,
    linear,
    relu,
    rms_norm,
    sigmoid,
    silu,
    softplus,
    tanh,
)


def _tensor() -> SparseTensor:
    return SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[-1.0, 2.0], [3.0, -4.0]], dtype=mx.float32),
    )


def test_linear_preserves_coordinate_identity() -> None:
    x = _tensor()
    weight = mx.array([[2.0, 3.0], [5.0, 7.0]], dtype=mx.float32)
    bias = mx.array([1.0, -1.0], dtype=mx.float32)

    out = linear(x, weight, bias)

    assert out.feats.tolist() == [[5.0, 8.0], [-5.0, -14.0]]
    assert out.coord_key == x.coord_key
    assert out.coord_manager is x.coord_manager
    assert out.coords is x.coords


def test_activation_feature_ops_match_expected_values() -> None:
    x = _tensor()

    assert relu(x).feats.tolist() == [[0.0, 2.0], [3.0, 0.0]]
    leak = cast(
        'list[list[float]]',
        leaky_relu(x, negative_slope=0.1).feats.tolist(),
    )
    assert leak[0][0] == pytest.approx(-0.1)
    assert leak[0][1] == pytest.approx(2.0)
    assert leak[1][0] == pytest.approx(3.0)
    assert leak[1][1] == pytest.approx(-0.4)
    assert tanh(x).coord_key == x.coord_key

    for out in [sigmoid(x), silu(x), softplus(x), gelu(x)]:
        assert out.feats.shape == x.feats.shape


def test_dropout_eval_is_identity_and_training_masks_features() -> None:
    x = _tensor()

    eval_out = dropout(x, training=False)
    train_out = dropout(x, p=0.5)

    assert eval_out.feats.tolist() == x.feats.tolist()
    assert eval_out.coord_key == x.coord_key
    assert train_out.coord_key == x.coord_key
    assert train_out.feats.shape == x.feats.shape


def test_normalization_feature_ops_apply_affine_parameters() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[1.0, 3.0], [3.0, 7.0]], dtype=mx.float32),
    )
    weight = mx.array([2.0, 3.0], dtype=mx.float32)
    bias = mx.array([0.5, -0.5], dtype=mx.float32)

    bn = batch_norm(
        x,
        weight=weight,
        bias=bias,
        mean=mx.array([2.0, 5.0], dtype=mx.float32),
        var=mx.array([1.0, 4.0], dtype=mx.float32),
        eps=1e-12,
    )
    ln = layer_norm(x, weight=weight, bias=bias, eps=1e-12)
    rms = rms_norm(x, weight=weight, eps=1e-12)

    assert bn.feats.tolist() == [[-1.5, -3.5], [2.5, 2.5]]
    assert ln.coord_key == x.coord_key
    assert rms.coord_key == x.coord_key
    assert ln.feats.shape == x.feats.shape
    assert rms.feats.shape == x.feats.shape


def test_feature_ops_reject_invalid_contracts() -> None:
    x = _tensor()

    with pytest.raises(ValueError, match='weight'):
        linear(x, mx.ones((2, 3), dtype=mx.float32))
    with pytest.raises(ValueError, match='bias'):
        linear(x, mx.ones((1, 2), dtype=mx.float32), mx.ones((2, 1)))
    with pytest.raises(ValueError, match='approximate'):
        gelu(x, approximate=cast('Any', 'fast'))
    with pytest.raises(ValueError, match='p must'):
        dropout(x, p=1.0)
    with pytest.raises(ValueError, match='eps'):
        layer_norm(x, eps=0)

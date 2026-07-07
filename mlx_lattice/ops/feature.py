from __future__ import annotations

from typing import Annotated, Literal, cast

import mlx.core as mx

from mlx_lattice.artifact.lowering import (
    array_operand,
    artifact_lowering,
    float_attribute,
    lattice_lowering,
    linear_weight_operand,
    optional_array_operand,
    str_attribute,
)
from mlx_lattice.core import QuantizedWeight, SparseTensor
from mlx_lattice.ops._quantized import quantized_matmul

GeluApprox = Literal['none', 'precise', 'tanh', 'fast']

__all__ = [
    'activation',
    'batch_norm',
    'batch_norm_features',
    'dropout',
    'gelu',
    'layer_norm',
    'layer_norm_features',
    'leaky_relu',
    'linear',
    'linear_features',
    'relu',
    'rms_norm',
    'rms_norm_features',
    'sigmoid',
    'silu',
    'softplus',
    'tanh',
]


@lattice_lowering(op='linear')
def linear_features(
    x: mx.array,
    weight: mx.array | QuantizedWeight,
    bias: mx.array | None = None,
) -> mx.array:
    """Apply a dense or quantized linear projection to a feature matrix.

    ``x`` must have shape ``(N, C_in)``. Dense weights use shape
    ``(C_out, C_in)`` and packed weights use ``QuantizedWeight`` with
    ``linear`` layout. Optional bias has shape ``(C_out,)``.
    """
    if x.ndim != 2:
        raise ValueError('linear_features expects a rank-2 feature tensor.')
    if isinstance(weight, QuantizedWeight):
        return _with_bias(quantized_matmul(x, weight), bias)
    _require_2d_weight(weight)
    if weight.shape[1] != x.shape[1]:
        raise ValueError('weight input channels must match x.shape[1].')
    return _with_bias(x @ weight.T, bias)


def linear(
    x: SparseTensor,
    weight: mx.array | QuantizedWeight,
    bias: mx.array | None = None,
) -> SparseTensor:
    """Apply a dense or quantized linear projection to sparse features.

    Coordinates are preserved. Dense weights use shape ``(C_out, C_in)`` and
    packed weights use ``QuantizedWeight`` with ``linear`` layout. Optional
    bias has shape ``(C_out,)``.
    """
    return x.replace(feats=linear_features(x.feats, weight, bias))


@artifact_lowering(op=linear_features)
def linear_from_artifact(
    input_value: Annotated[mx.array, array_operand(0)],
    weight: Annotated[
        mx.array | QuantizedWeight,
        linear_weight_operand(1, input='input_value'),
    ],
    bias: Annotated[mx.array | None, optional_array_operand(2)],
) -> mx.array:
    """Lower lattice.linear artifact ops through dense feature tensors."""

    return linear_features(input_value, weight, bias)


ActivationKind = Literal[
    'relu',
    'sigmoid',
    'gelu',
    'silu',
    'leaky_relu',
    'tanh',
    'softplus',
]


@lattice_lowering
def activation(
    x: mx.array,
    *,
    kind: ActivationKind,
    approximate: GeluApprox = 'none',
    alpha: float = 0.01,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> mx.array:
    """Apply a dense feature activation to a rank-2 feature tensor."""
    if x.ndim != 2:
        raise ValueError('activation expects a rank-2 feature tensor.')
    if kind == 'relu':
        return mx.maximum(x, 0)
    if kind == 'sigmoid':
        return mx.sigmoid(x)
    if kind == 'gelu':
        return _gelu_features(x, approximate=approximate)
    if kind == 'silu':
        return x * mx.sigmoid(x)
    if kind == 'leaky_relu':
        slope = mx.array(float(alpha), dtype=x.dtype)
        return mx.where(x >= 0, x, x * slope)
    if kind == 'tanh':
        return mx.tanh(x)
    if kind == 'softplus':
        if beta <= 0:
            raise ValueError('beta must be positive.')
        scaled = x * beta
        return mx.where(
            scaled > threshold,
            x,
            mx.log(1 + mx.exp(scaled)) / beta,
        )
    raise ValueError(
        "kind must be 'relu', 'sigmoid', 'gelu', 'silu', "
        "'leaky_relu', 'tanh', or 'softplus'."
    )


@artifact_lowering(op=activation)
def activation_from_artifact(
    x: Annotated[mx.array, array_operand(0)],
    *,
    kind: Annotated[str, str_attribute()],
    approximate: Annotated[str, str_attribute()],
    alpha: Annotated[float, float_attribute()],
    beta: Annotated[float, float_attribute()],
    threshold: Annotated[float, float_attribute()],
) -> mx.array:
    """Lower lattice.activation artifact ops through ``activation``."""

    return activation(
        x,
        kind=_activation_kind(kind),
        approximate=_gelu_approx(approximate),
        alpha=alpha,
        beta=beta,
        threshold=threshold,
    )


def relu(x: SparseTensor) -> SparseTensor:
    """Apply ReLU to sparse features while preserving coordinates."""
    return x.replace(feats=activation(x.feats, kind='relu'))


def sigmoid(x: SparseTensor) -> SparseTensor:
    """Apply sigmoid to sparse features while preserving coordinates."""
    return x.replace(feats=activation(x.feats, kind='sigmoid'))


def gelu(
    x: SparseTensor,
    *,
    approximate: GeluApprox = 'none',
) -> SparseTensor:
    """Apply GELU to sparse features while preserving coordinates.

    ``approximate`` accepts ``'none'``/``'precise'`` for the erf formula,
    ``'tanh'`` for the tanh approximation, or ``'fast'`` for the sigmoid-based
    approximation.
    """
    return x.replace(
        feats=activation(x.feats, kind='gelu', approximate=approximate)
    )


def _gelu_features(
    feats: mx.array,
    *,
    approximate: GeluApprox,
) -> mx.array:
    if approximate in ('none', 'precise'):
        scale = mx.array(0.5, dtype=feats.dtype)
        root_half = mx.array(0.7071067811865476, dtype=feats.dtype)
        return scale * feats * (1 + mx.erf(feats * root_half))
    if approximate == 'tanh':
        coeff = mx.array(0.044715, dtype=feats.dtype)
        scale = mx.array(0.7978845608028654, dtype=feats.dtype)
        return (
            0.5 * feats * (1 + mx.tanh(scale * (feats + coeff * feats**3)))
        )
    if approximate == 'fast':
        return feats * mx.sigmoid(1.702 * feats)
    raise ValueError(
        "approximate must be 'none', 'precise', 'tanh', or 'fast'."
    )


def _activation_kind(value: str) -> ActivationKind:
    if value not in (
        'relu',
        'sigmoid',
        'gelu',
        'silu',
        'leaky_relu',
        'tanh',
        'softplus',
    ):
        raise ValueError(f'unsupported activation kind: {value}')
    return cast(ActivationKind, value)


def _gelu_approx(value: str) -> GeluApprox:
    if value not in ('none', 'precise', 'tanh', 'fast'):
        raise ValueError(f'unsupported GELU approximation: {value}')
    return cast(GeluApprox, value)


def silu(x: SparseTensor) -> SparseTensor:
    """Apply SiLU/Swish to sparse features while preserving coordinates."""
    return x.replace(feats=activation(x.feats, kind='silu'))


def leaky_relu(
    x: SparseTensor,
    *,
    negative_slope: float = 0.01,
) -> SparseTensor:
    """Apply leaky ReLU to sparse features while preserving coordinates."""
    return x.replace(
        feats=activation(
            x.feats,
            kind='leaky_relu',
            alpha=negative_slope,
        )
    )


def tanh(x: SparseTensor) -> SparseTensor:
    """Apply hyperbolic tangent to sparse features while preserving coordinates."""
    return x.replace(feats=activation(x.feats, kind='tanh'))


def softplus(
    x: SparseTensor,
    *,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> SparseTensor:
    """Apply numerically thresholded softplus to sparse features.

    Values above ``threshold`` in the scaled domain return the input directly
    to avoid unnecessary exponential work.
    """
    if beta <= 0:
        raise ValueError('beta must be positive.')
    return x.replace(
        feats=activation(
            x.feats,
            kind='softplus',
            beta=beta,
            threshold=threshold,
        )
    )


@lattice_lowering(op='batch_norm')
def batch_norm_features(
    x: mx.array,
    scale: mx.array,
    bias: mx.array,
    mean: mx.array,
    var: mx.array,
    *,
    eps: float = 1e-5,
) -> mx.array:
    """Apply frozen-stat batch normalization to rank-2 feature rows."""
    if x.ndim != 2:
        raise ValueError('batch_norm_features expects rank-2 features.')
    if eps <= 0:
        raise ValueError('eps must be positive.')
    channels = int(x.shape[1])
    _require_channel_vector(scale, channels, 'scale')
    _require_channel_vector(bias, channels, 'bias')
    _require_channel_vector(mean, channels, 'mean')
    _require_channel_vector(var, channels, 'var')
    return (x - mean) * mx.rsqrt(var + eps) * scale + bias


@artifact_lowering(op=batch_norm_features)
def batch_norm_from_artifact(
    x: Annotated[mx.array, array_operand(0)],
    scale: Annotated[mx.array, array_operand(1)],
    bias: Annotated[mx.array, array_operand(2)],
    mean: Annotated[mx.array, array_operand(3)],
    var: Annotated[mx.array, array_operand(4)],
    *,
    eps: Annotated[float, float_attribute()],
) -> mx.array:
    """Lower lattice.batch_norm artifact ops through dense features."""

    return batch_norm_features(
        x,
        scale,
        bias,
        mean,
        var,
        eps=eps,
    )


def dropout(
    x: SparseTensor,
    *,
    p: float = 0.5,
    training: bool = True,
) -> SparseTensor:
    """Apply inverted dropout to sparse features during training.

    Coordinates are preserved. When ``training`` is false or ``p`` is zero, the
    feature matrix is returned unchanged inside a new sparse wrapper.
    """
    if p < 0 or p >= 1:
        raise ValueError('p must satisfy 0 <= p < 1.')
    if not training or p == 0:
        return x.replace(feats=x.feats)
    keep = 1.0 - p
    mask = mx.random.bernoulli(p=keep, shape=x.feats.shape)
    return x.replace(feats=x.feats * mask.astype(x.feats.dtype) / keep)


def batch_norm(
    x: SparseTensor,
    *,
    weight: mx.array | None = None,
    bias: mx.array | None = None,
    mean: mx.array | None = None,
    var: mx.array | None = None,
    eps: float = 1e-5,
) -> SparseTensor:
    """Apply per-channel batch normalization to sparse features.

    If ``mean`` or ``var`` is omitted, statistics are computed from active
    feature rows. Optional affine ``weight`` and ``bias`` have shape ``(C,)``.
    """
    if eps <= 0:
        raise ValueError('eps must be positive.')
    mean = mx.mean(x.feats, axis=0) if mean is None else mean
    var = mx.var(x.feats, axis=0) if var is None else var
    scale = (
        mx.ones((x.channels,), dtype=x.feats.dtype)
        if weight is None
        else weight
    )
    offset = (
        mx.zeros((x.channels,), dtype=x.feats.dtype)
        if bias is None
        else bias
    )
    return x.replace(
        feats=batch_norm_features(
            x.feats,
            scale,
            offset,
            mean,
            var,
            eps=eps,
        )
    )


@lattice_lowering(op='layer_norm')
def layer_norm_features(
    x: mx.array,
    scale: mx.array,
    bias: mx.array,
    *,
    eps: float = 1e-5,
) -> mx.array:
    """Apply layer normalization to rank-2 feature rows."""
    if x.ndim != 2:
        raise ValueError('layer_norm_features expects rank-2 features.')
    if eps <= 0:
        raise ValueError('eps must be positive.')
    channels = int(x.shape[1])
    _require_channel_vector(scale, channels, 'scale')
    _require_channel_vector(bias, channels, 'bias')
    return mx.fast.layer_norm(x, scale, bias, eps)


@artifact_lowering(op=layer_norm_features)
def layer_norm_from_artifact(
    x: Annotated[mx.array, array_operand(0)],
    scale: Annotated[mx.array, array_operand(1)],
    bias: Annotated[mx.array, array_operand(2)],
    *,
    eps: Annotated[float, float_attribute()],
) -> mx.array:
    """Lower lattice.layer_norm artifact ops through dense features."""

    return layer_norm_features(
        x,
        scale,
        bias,
        eps=eps,
    )


def layer_norm(
    x: SparseTensor,
    *,
    weight: mx.array | None = None,
    bias: mx.array | None = None,
    eps: float = 1e-5,
) -> SparseTensor:
    """Apply layer normalization independently to each sparse row."""
    if eps <= 0:
        raise ValueError('eps must be positive.')
    scale = (
        mx.ones((x.channels,), dtype=x.feats.dtype)
        if weight is None
        else weight
    )
    offset = (
        mx.zeros((x.channels,), dtype=x.feats.dtype)
        if bias is None
        else bias
    )
    return x.replace(
        feats=layer_norm_features(x.feats, scale, offset, eps=eps)
    )


@lattice_lowering(op='rms_norm')
def rms_norm_features(
    x: mx.array,
    scale: mx.array,
    *,
    eps: float = 1e-5,
) -> mx.array:
    """Apply RMS normalization to rank-2 feature rows."""
    if x.ndim != 2:
        raise ValueError('rms_norm_features expects rank-2 features.')
    if eps <= 0:
        raise ValueError('eps must be positive.')
    _require_channel_vector(scale, int(x.shape[1]), 'scale')
    return mx.fast.rms_norm(x, scale, eps)


@artifact_lowering(op=rms_norm_features)
def rms_norm_from_artifact(
    x: Annotated[mx.array, array_operand(0)],
    scale: Annotated[mx.array, array_operand(1)],
    *,
    eps: Annotated[float, float_attribute()],
) -> mx.array:
    """Lower lattice.rms_norm artifact ops through dense features."""

    return rms_norm_features(x, scale, eps=eps)


def rms_norm(
    x: SparseTensor,
    *,
    weight: mx.array | None = None,
    eps: float = 1e-5,
) -> SparseTensor:
    """Apply RMS normalization independently to each sparse row."""
    if eps <= 0:
        raise ValueError('eps must be positive.')
    scale = (
        mx.ones((x.channels,), dtype=x.feats.dtype)
        if weight is None
        else weight
    )
    return x.replace(feats=rms_norm_features(x.feats, scale, eps=eps))


# MARK: - helpers


def _affine(
    feats: mx.array,
    *,
    weight: mx.array | None,
    bias: mx.array | None,
) -> mx.array:
    if weight is not None:
        _require_channel_vector(weight, int(feats.shape[1]), 'weight')
        feats = feats * weight
    return _with_bias(feats, bias)


def _with_bias(feats: mx.array, bias: mx.array | None) -> mx.array:
    if bias is None:
        return feats
    _require_channel_vector(bias, int(feats.shape[1]), 'bias')
    return feats + bias


def _require_2d_weight(weight: mx.array) -> None:
    if weight.ndim != 2:
        raise ValueError('weight must have shape (C_out, C_in).')


def _require_channel_vector(
    value: mx.array,
    channels: int,
    name: str,
) -> None:
    if value.ndim != 1 or value.shape[0] != channels:
        raise ValueError(f'{name} must have shape ({channels},).')

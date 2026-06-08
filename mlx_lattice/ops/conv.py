from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice.core import KernelRelation, KernelSpec, SparseTensor
from mlx_lattice.core.types import Triple
from mlx_lattice.ops.exec import execute_spmm
from mlx_lattice.ops.maps import (
    generative_kernel_relation,
    kernel_relation,
    transposed_kernel_relation,
)

__all__ = [
    'conv3d',
    'conv_transpose3d',
    'generative_conv_transpose3d',
    'subm_conv3d',
]


def conv3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if spec.is_pointwise:
        return x.replace(
            feats=_with_bias(_pointwise_features(x, weight), bias)
        )

    relation = kernel_relation(
        x,
        kernel_size=spec.size,
        stride=spec.stride,
        padding=spec.padding,
        dilation=spec.dilation,
    )
    return _mapped_conv(
        x,
        weight,
        bias,
        relation,
        output_stride=_mul_stride(x.stride, spec.stride),
    )


def subm_conv3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 3,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=1,
        padding=0,
        dilation=dilation,
    )
    _require_odd_kernel(spec.size, 'subm_conv3d')
    if spec.size == (1, 1, 1) and spec.dilation == (1, 1, 1):
        return x.replace(
            feats=_with_bias(_pointwise_features(x, weight), bias)
        )

    relation = kernel_relation(
        x,
        kernel_size=spec.size,
        stride=1,
        padding=0,
        dilation=spec.dilation,
    )
    feats = _spmm_features(x, weight, bias, relation)
    return x.replace(feats=feats)


def conv_transpose3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    relation = transposed_kernel_relation(
        x,
        kernel_size=spec.size,
        stride=spec.stride,
        padding=spec.padding,
        dilation=spec.dilation,
    )
    return _mapped_conv(
        x,
        weight,
        bias,
        relation,
        output_stride=_div_stride(x.stride, spec.stride),
    )


def generative_conv_transpose3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> SparseTensor:
    spec = KernelSpec(size=kernel_size, stride=stride)
    relation = generative_kernel_relation(
        x,
        kernel_size=spec.size,
        stride=spec.stride,
    )
    return _mapped_conv(
        x,
        weight,
        bias,
        relation,
        output_stride=_div_stride(x.stride, spec.stride),
    )


# MARK: - execution policy


def _mapped_conv(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None,
    relation: KernelRelation,
    *,
    output_stride: Triple,
) -> SparseTensor:
    if relation.out_coords is None:
        raise ValueError(
            'convolution relations must define output coordinates.'
        )
    feats = _spmm_features(x, weight, bias, relation)
    return SparseTensor(
        relation.out_coords,
        feats,
        stride=output_stride,
        coord_manager=x.coord_manager,
    )


def _spmm_features(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None,
    relation: KernelRelation,
) -> mx.array:
    _validate_feature_dtype(x.feats, weight)
    mapped_weight = _mapped_weight(x, weight, relation)
    return _with_bias(execute_spmm(x.feats, mapped_weight, relation), bias)


def _pointwise_features(x: SparseTensor, weight: mx.array) -> mx.array:
    _validate_feature_dtype(x.feats, weight)
    matrix = _pointwise_weight_matrix(x, weight)
    return x.feats @ matrix.T


# MARK: - validation


def _validate_feature_dtype(feats: mx.array, weight: mx.array) -> None:
    if feats.dtype != mx.float32 or weight.dtype != mx.float32:
        raise ValueError('convolution currently supports float32 tensors.')


def _pointwise_weight_matrix(x: SparseTensor, weight: mx.array) -> mx.array:
    if weight.ndim == 2:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight
    if (
        weight.ndim == 5
        and weight.shape[1] == 1
        and weight.shape[2] == 1
        and weight.shape[3] == 1
    ):
        if weight.shape[4] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight[:, 0, 0, 0, :]
    if weight.ndim == 3 and weight.shape[0] == 1:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight[0].T
    raise ValueError(
        'pointwise weight must have shape (C_out, C_in), '
        '(C_out, 1, 1, 1, C_in), or (1, C_in, C_out).'
    )


def _mapped_weight(
    x: SparseTensor,
    weight: mx.array,
    relation: KernelRelation,
) -> mx.array:
    if weight.ndim == 3:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        if (
            relation.n_kernels is not None
            and weight.shape[0] != relation.n_kernels
        ):
            raise ValueError(
                'weight kernel rows must match the kernel relation.'
            )
        return weight

    if weight.ndim != 5:
        raise ValueError(
            'mapped convolution weight must have shape (K, C_in, C_out) '
            'or (C_out, Kx, Ky, Kz, C_in).'
        )
    if weight.shape[4] != x.channels:
        raise ValueError('weight input channels must match x.channels.')
    kernel_rows = int(weight.shape[1] * weight.shape[2] * weight.shape[3])
    if relation.n_kernels is not None and kernel_rows != relation.n_kernels:
        raise ValueError(
            'weight kernel rows must match the kernel relation.'
        )
    out_channels = int(weight.shape[0])
    return weight.reshape(out_channels, kernel_rows, x.channels).transpose(
        1, 2, 0
    )


def _with_bias(feats: mx.array, bias: mx.array | None) -> mx.array:
    if bias is None:
        return feats
    if bias.ndim != 1:
        raise ValueError('bias must have shape (C_out,).')
    if bias.shape[0] != feats.shape[1]:
        raise ValueError('bias channels must match output channels.')
    if bias.dtype != feats.dtype:
        raise ValueError('bias dtype must match output features.')
    return feats + bias


def _require_odd_kernel(values: Triple, op_name: str) -> None:
    if any(value % 2 == 0 for value in values):
        raise ValueError(f'{op_name} requires odd kernel_size values.')


def _mul_stride(lhs: Triple, rhs: Triple) -> Triple:
    return (lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2])


def _div_stride(lhs: Triple, rhs: Triple) -> Triple:
    out = []
    for left, right in zip(lhs, rhs, strict=True):
        if left % right != 0:
            raise ValueError(
                'transpose stride must divide the input tensor stride.'
            )
        out.append(left // right)
    return (out[0], out[1], out[2])

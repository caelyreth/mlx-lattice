from __future__ import annotations

import mlx.core as mx

from mlx_lattice.core.quantized import QuantizedWeight, dequantize_weight
from mlx_lattice.ops._projection import (
    feature_projection,
    should_use_precise_small_fp32_projection,
)


def quantized_matmul(
    feats: mx.array,
    weight: QuantizedWeight,
) -> mx.array:
    """Apply affine packed-weight matrix multiplication to feature rows."""
    if weight.kernel_size != (1, 1, 1):
        raise ValueError('quantized matmul requires a pointwise weight.')
    if feats.ndim != 2 or feats.shape[1] != weight.in_channels:
        raise ValueError(
            'features must have shape (N, quantized_weight.in_channels).'
        )
    if feats.dtype not in (mx.float16, mx.float32):
        raise ValueError('features must be float16 or float32.')
    if should_use_precise_small_fp32_projection(
        feats,
        in_channels=weight.in_channels,
        out_channels=weight.out_channels,
        weight_dtype=mx.float32,
    ):
        return feature_projection(
            feats,
            _pointwise_matrix(dequantize_weight(weight)).astype(mx.float32),
        )
    output_dtype = feats.dtype
    feats = feats.astype(weight.scales.dtype)
    if weight.storage_in_channels != weight.in_channels:
        feats = mx.pad(
            feats,
            [
                (0, 0),
                (0, weight.storage_in_channels - weight.in_channels),
            ],
        )
    out = mx.quantized_matmul(
        feats,
        weight.weight[0],
        weight.scales[0],
        weight.biases[0],
        transpose=True,
        group_size=weight.group_size,
        bits=weight.bits,
        mode='affine',
    )
    return out.astype(output_dtype)


def _pointwise_matrix(weight: mx.array) -> mx.array:
    if weight.ndim == 2:
        return weight
    if weight.ndim == 3:
        return weight[0].T
    if weight.ndim == 5:
        return weight[:, 0, 0, 0, :]
    raise ValueError('pointwise quantized weight has an invalid layout.')

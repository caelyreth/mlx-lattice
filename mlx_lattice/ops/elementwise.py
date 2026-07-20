from __future__ import annotations

import mlx.core as mx
from lattice_contract.dialect import elementwise, embedding_lookup, softmax

from mlx_lattice.artifact.lowering import artifact_lowering


@artifact_lowering(op=embedding_lookup)
def embedding_lookup_from_artifact(
    input: mx.array,
    weight: mx.array,
) -> mx.array:
    """Gather dense embedding rows for an IR v2 artifact."""
    if input.dtype not in (mx.int32, mx.int64):
        raise TypeError(
            'lattice.embedding_lookup indices must be integers.'
        )
    if weight.ndim != 2:
        raise ValueError('lattice.embedding_lookup weight must be rank 2.')
    return mx.take(weight, input.astype(mx.int32), axis=0)


@artifact_lowering(op=elementwise)
def elementwise_from_artifact(
    input: mx.array,
    *,
    kind: str,
) -> mx.array:
    """Execute the deterministic IR v2 scalar transforms in FP32."""
    value = input.astype(mx.float32)
    if kind == 'exp':
        return mx.exp(value)
    if kind == 'round':
        lower = mx.floor(value)
        fraction = value - lower
        odd_lower = mx.not_equal(
            mx.bitwise_and(lower.astype(mx.int32), 1), 0
        )
        increment = mx.logical_or(
            mx.greater(fraction, 0.5),
            mx.logical_and(mx.equal(fraction, 0.5), odd_lower),
        )
        return lower + increment.astype(mx.float32)
    raise ValueError(
        'unsupported lattice.elementwise kind: '
        f'{kind!r}; expected exp or round.'
    )


@artifact_lowering(op=softmax)
def softmax_from_artifact(
    input: mx.array,
    *,
    axis: int,
) -> mx.array:
    """Execute IR v2 FP32 subtract-max softmax on the final dimension."""
    if axis not in (-1, input.ndim - 1):
        raise ValueError(
            'IR v2 lattice.softmax only supports the last axis.'
        )
    value = input.astype(mx.float32)
    shifted = value - mx.max(value, axis=-1, keepdims=True)
    numerator = mx.exp(shifted)
    return numerator / mx.sum(numerator, axis=-1, keepdims=True)

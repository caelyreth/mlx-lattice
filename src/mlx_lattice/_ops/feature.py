from __future__ import annotations

import mlx.core as mx

from mlx_lattice.tensor import SparseTensor


def linear(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
) -> SparseTensor:
    if weight.ndim != 2:
        raise ValueError('weight must have shape (Cout, Cin).')
    if weight.shape[1] != x.channels:
        raise ValueError(
            'weight input channels must match tensor features.'
        )
    feats = x.feats @ mx.swapaxes(weight, 0, 1)
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError('bias must have shape (Cout,).')
        feats = feats + bias
    return x.replace(feats=feats)


def relu(x: SparseTensor) -> SparseTensor:
    return x.replace(feats=mx.maximum(x.feats, 0))


def sigmoid(x: SparseTensor) -> SparseTensor:
    return x.replace(feats=mx.sigmoid(x.feats))

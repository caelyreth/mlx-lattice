from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice.tensor import SparseTensor


def sparse_collate(
    coords: Sequence[mx.array],
    feats: Sequence[mx.array],
) -> SparseTensor:
    if len(coords) != len(feats):
        raise ValueError('coords and feats batch sizes must match.')

    batched_coords = []
    for batch, values in enumerate(coords):
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError('collated coords must have shape (N, 3).')
        batch_col = mx.full((values.shape[0], 1), batch, dtype=values.dtype)
        batched_coords.append(mx.concatenate([batch_col, values], axis=1))
    return SparseTensor(
        mx.concatenate(batched_coords, axis=0),
        mx.concatenate(list(feats), axis=0),
        batch_counts=tuple(int(values.shape[0]) for values in coords),
    )


def cat(tensors: Sequence[SparseTensor]) -> SparseTensor:
    if not tensors:
        raise ValueError('expected at least one sparse tensor.')
    first = tensors[0]
    for tensor in tensors[1:]:
        if not first.same_coords(tensor):
            raise ValueError('sparse tensor coordinates must match.')
    return first.replace(
        feats=mx.concatenate([tensor.feats for tensor in tensors], axis=1)
    )


def prune(x: SparseTensor, rows: mx.array) -> SparseTensor:
    if rows.ndim != 1:
        raise ValueError('rows must have shape (M,).')
    rows = rows.astype(mx.int32)
    return SparseTensor(
        mx.take(x.coords, rows, axis=0),
        mx.take(x.feats, rows, axis=0),
        x.stride,
    )


def topk_rows(
    x: SparseTensor,
    counts: Sequence[int],
    *,
    rho: float = 1.0,
) -> mx.array:
    if rho <= 0:
        raise ValueError('rho must be positive.')

    selected = []
    start = 0
    row_counts = x.batch_counts
    if row_counts is None:
        row_counts = tuple(int(rows.shape[0]) for rows in x.batch_rows)
    if len(counts) != len(row_counts):
        raise ValueError('counts must match the batch count.')

    for keep, row_count in zip(counts, row_counts, strict=True):
        stop = start + int(row_count)
        if stop > x.n_points:
            raise ValueError(
                'batch row counts exceed sparse tensor row count.'
            )
        rows = mx.arange(start, stop, dtype=mx.int32)
        start = stop
        k = min(int(keep * rho), int(rows.shape[0]))
        if k <= 0:
            continue
        scores = mx.take(x.feats[:, 0], rows, axis=0)
        order = mx.argsort(scores)
        selected.append(mx.take(rows, order[-k:], axis=0))
    if start != x.n_points:
        raise ValueError('counts must cover all sparse tensor rows.')
    if not selected:
        return mx.array([], dtype=mx.int32)
    return mx.concatenate(selected, axis=0)

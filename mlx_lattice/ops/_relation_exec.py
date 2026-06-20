from __future__ import annotations

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.relations import KernelRelation


def sparse_conv_features_from_relation(
    feats: mx.array,
    weight: mx.array,
    relation: KernelRelation,
) -> mx.array:
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )
    input_csr = relation.input_csr
    kernel_csr = relation.kernel_csr
    if input_csr.edge_ids is None or kernel_csr.edge_ids is None:
        raise ValueError('kernel relation is missing grouped CSR views.')
    return ext.sparse_conv_features(
        feats,
        weight,
        relation.edges.in_rows,
        relation.edges.out_rows,
        relation.edges.kernel_ids,
        relation.counts,
        relation.output_csr.row_offsets,
        input_csr.row_offsets,
        input_csr.edge_ids,
        kernel_csr.row_offsets,
        kernel_csr.edge_ids,
        relation.n_out_capacity,
        relation.n_kernels,
    )


def sparse_pool_features_from_relation(
    feats: mx.array,
    relation: KernelRelation,
    *,
    input_exclusive: bool,
    mode: str,
) -> mx.array:
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )
    return ext.sparse_pool_features(
        feats,
        relation.edges.in_rows,
        relation.edges.out_rows,
        relation.edges.kernel_ids,
        relation.output_csr.row_offsets,
        relation.counts,
        input_exclusive,
        mode,
        relation.n_out_capacity,
        relation.n_kernels,
    )

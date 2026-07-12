from __future__ import annotations

import math
from collections.abc import Sequence

import mlx.core as mx
from lattice_contract import indexed_kernel_offsets, sparse_kernel_offsets

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import validate_coords
from mlx_lattice.core.relations import (
    KernelRelation,
    KernelSpec,
    NeighborRelation,
    RelationImplicitGemmView,
)
from mlx_lattice.core.relations.views import RelationKind
from mlx_lattice.core.types import Triple, triple

type NativeKernelRelation = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]
type NativeNeighborRelation = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]
type NativeImplicitGemmView = tuple[mx.array, mx.array]


def build_relation_implicit_gemm_view(
    relation: KernelRelation,
) -> RelationImplicitGemmView:
    """Build the dense output-to-input map used by implicit-GEMM kernels."""
    contract = relation.contract
    if contract.kind not in ('forward', 'target', 'submanifold'):
        raise ValueError(
            'implicit GEMM view currently supports forward, target, and '
            'submanifold relations.'
        )
    if (
        contract.source_coords is None
        or contract.source_active_rows is None
        or contract.out_coords is None
    ):
        raise ValueError(
            'kernel relation is missing coordinate context for implicit GEMM.'
        )
    output_active = (
        contract.target_active_rows
        if contract.kind in ('target', 'submanifold')
        else contract.out_count
    )
    if output_active is None:
        raise ValueError(
            'kernel relation is missing output activity for implicit GEMM.'
        )
    offsets = _offset_array(contract.kernel_offsets)
    out_in_map, row_masks = ext.build_relation_implicit_gemm_view(
        contract.source_coords,
        contract.source_active_rows,
        contract.out_coords,
        output_active,
        offsets,
        contract.stride,
        contract.padding,
    )
    return RelationImplicitGemmView(out_in_map, row_masks=row_masks)


def build_target_transposed_implicit_gemm_view(
    source_coords: mx.array,
    target_coords: mx.array,
    *,
    source_active_rows: mx.array,
    target_active_rows: mx.array,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> RelationImplicitGemmView:
    """Build a target-row by kernel map for transposed sparse geometry."""
    validate_coords(source_coords)
    validate_coords(target_coords)
    _require_matching_coord_dtype(source_coords, target_coords)
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    offsets = _offset_array(
        _indexed_kernel_offsets(spec.size, spec.dilation)
    )
    out_in_map, row_masks = ext.build_target_transposed_implicit_gemm_view(
        source_coords,
        source_active_rows,
        target_coords,
        target_active_rows,
        offsets,
        spec.stride,
        spec.padding,
    )
    return RelationImplicitGemmView(out_in_map, row_masks=row_masks)


def build_target_kernel_relation(
    coords: mx.array,
    target_coords: mx.array,
    *,
    active_rows: mx.array | None = None,
    target_active_rows: mx.array | None = None,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    """Build a sparse kernel relation from source coords to target coords."""
    validate_coords(coords)
    validate_coords(target_coords)
    _require_matching_coord_dtype(coords, target_coords)
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    offsets = kernel_offsets(spec.size, spec.dilation)
    source_active = _active_rows(active_rows, coords)
    target_active = _active_rows(target_active_rows, target_coords)
    native = ext.build_target_kernel_relation(
        coords,
        source_active,
        target_coords,
        target_active,
        spec.size,
        spec.stride,
        spec.padding,
        spec.dilation,
    )
    return _kernel_relation_from_native(
        native,
        offsets=offsets,
        in_capacity=int(coords.shape[0]),
        source_coords=coords,
        source_active_rows=source_active,
        target_coords=target_coords,
        target_active_rows=target_active,
        stride=spec.stride,
        padding=spec.padding,
        kind='target',
    )


def kernel_offsets(
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> tuple[Triple, ...]:
    """Enumerate spatial offsets for a dense 3D kernel footprint."""
    kernel = triple(kernel_size, name='kernel_size')
    rate = triple(dilation, name='dilation')
    _require_positive(kernel, 'kernel_size')
    _require_positive(rate, 'dilation')

    return sparse_kernel_offsets(kernel, rate)


def _indexed_kernel_offsets(
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> tuple[Triple, ...]:
    kernel = triple(kernel_size, name='kernel_size')
    rate = triple(dilation, name='dilation')
    _require_positive(kernel, 'kernel_size')
    _require_positive(rate, 'dilation')
    return indexed_kernel_offsets(kernel, rate)


def build_kernel_relation(
    coords: mx.array,
    active_rows: mx.array | None = None,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    """Build a forward sparse convolution/pooling relation."""
    validate_coords(coords)
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    offsets = kernel_offsets(spec.size, spec.dilation)
    source_active = _active_rows(active_rows, coords)
    native = ext.build_kernel_relation(
        coords,
        source_active,
        spec.size,
        spec.stride,
        spec.padding,
        spec.dilation,
    )
    return _kernel_relation_from_native(
        native,
        offsets=offsets,
        in_capacity=int(coords.shape[0]),
        source_coords=coords,
        source_active_rows=source_active,
        stride=spec.stride,
        padding=spec.padding,
        kind='forward',
    )


def build_submanifold_kernel_relation(
    coords: mx.array,
    active_rows: mx.array | None = None,
    kernel_size: int | Sequence[int] = 3,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    """Build a submanifold relation whose output support is ``coords``."""
    validate_coords(coords)
    spec = KernelSpec(
        size=kernel_size,
        stride=1,
        padding=0,
        dilation=dilation,
    )
    if not spec.is_centered_submanifold:
        raise ValueError(
            'submanifold relations require odd kernels, stride=1, '
            'padding=0, and positive dilation.'
        )
    offsets = kernel_offsets(spec.size, spec.dilation)
    source_active = _active_rows(active_rows, coords)
    native = ext.build_submanifold_kernel_relation(
        coords,
        source_active,
        spec.size,
        spec.dilation,
    )
    return _kernel_relation_from_native(
        native,
        offsets=offsets,
        in_capacity=int(coords.shape[0]),
        out_coords=coords,
        source_coords=coords,
        source_active_rows=source_active,
        target_coords=coords,
        target_active_rows=source_active,
        stride=spec.stride,
        padding=spec.padding,
        kind='submanifold',
    )


def build_generative_relation(
    coords: mx.array,
    active_rows: mx.array | None = None,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> KernelRelation:
    """Build a generative transpose-convolution relation."""
    validate_coords(coords)
    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    _require_positive(kernel, 'kernel_size')
    _require_positive(step, 'stride')

    offsets = _indexed_kernel_offsets(kernel)
    source_active = _active_rows(active_rows, coords)
    native = ext.build_generative_relation(
        coords,
        source_active,
        kernel,
        step,
    )
    return _kernel_relation_from_native(
        native,
        offsets=offsets,
        in_capacity=int(coords.shape[0]),
        source_coords=coords,
        source_active_rows=source_active,
        stride=step,
        kind='generative',
    )


def build_transposed_kernel_relation(
    coords: mx.array,
    active_rows: mx.array | None = None,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    """Build a sparse transpose-convolution relation."""
    validate_coords(coords)
    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    pad = triple(padding, name='padding')
    rate = triple(dilation, name='dilation')
    _require_positive(kernel, 'kernel_size')
    _require_positive(step, 'stride')
    _require_nonnegative(pad, 'padding')
    _require_positive(rate, 'dilation')

    offsets = _indexed_kernel_offsets(kernel, rate)
    source_active = _active_rows(active_rows, coords)
    native = ext.build_transposed_kernel_relation(
        coords,
        source_active,
        kernel,
        step,
        pad,
        rate,
    )
    return _kernel_relation_from_native(
        native,
        offsets=offsets,
        in_capacity=int(coords.shape[0]),
        source_coords=coords,
        source_active_rows=source_active,
        stride=step,
        padding=pad,
        kind='transposed',
    )


def build_knn_relation(
    source_coords: mx.array,
    query_coords: mx.array | None = None,
    *,
    source_active_rows: mx.array | None = None,
    query_active_rows: mx.array | None = None,
    k: int,
) -> NeighborRelation:
    """Build a k-nearest-neighbor relation between source and query coords."""
    query_coords = source_coords if query_coords is None else query_coords
    source_active_rows = _active_rows(source_active_rows, source_coords)
    query_active_rows = (
        source_active_rows
        if query_active_rows is None and query_coords is source_coords
        else _active_rows(query_active_rows, query_coords)
    )
    validate_coords(source_coords)
    validate_coords(query_coords)
    _require_matching_coord_dtype(source_coords, query_coords)
    neighbor_count = _positive_int(k, 'k')
    native = ext.build_knn_relation(
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        neighbor_count,
    )
    return _neighbor_relation_from_native(
        native,
        query_capacity=int(query_coords.shape[0]),
        source_capacity=int(source_coords.shape[0]),
        max_neighbors=neighbor_count,
    )


def build_radius_relation(
    source_coords: mx.array,
    query_coords: mx.array | None = None,
    *,
    source_active_rows: mx.array | None = None,
    query_active_rows: mx.array | None = None,
    radius: float,
    max_neighbors: int | None = None,
) -> NeighborRelation:
    """Build a radius-neighborhood relation between source and query coords."""
    query_coords = source_coords if query_coords is None else query_coords
    source_active_rows = _active_rows(source_active_rows, source_coords)
    query_active_rows = (
        source_active_rows
        if query_active_rows is None and query_coords is source_coords
        else _active_rows(query_active_rows, query_coords)
    )
    validate_coords(source_coords)
    validate_coords(query_coords)
    _require_matching_coord_dtype(source_coords, query_coords)
    radius_value = _nonnegative_float(radius, 'radius')
    neighbor_count = (
        0
        if max_neighbors is None
        else _positive_int(max_neighbors, 'max_neighbors')
    )
    native = ext.build_radius_relation(
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        radius_value,
        neighbor_count,
    )
    return _neighbor_relation_from_native(
        native,
        query_capacity=int(query_coords.shape[0]),
        source_capacity=int(source_coords.shape[0]),
        max_neighbors=(
            _radius_neighbor_capacity(radius_value)
            if max_neighbors is None
            else neighbor_count
        ),
    )


# MARK: - views


def _kernel_relation_from_native(
    native: NativeKernelRelation,
    *,
    offsets: tuple[Triple, ...],
    in_capacity: int,
    out_coords: mx.array | None = None,
    source_coords: mx.array,
    source_active_rows: mx.array,
    target_coords: mx.array | None = None,
    target_active_rows: mx.array | None = None,
    stride: Triple = (1, 1, 1),
    padding: Triple = (0, 0, 0),
    kind: RelationKind = 'forward',
) -> KernelRelation:
    (
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        native_out_coords,
        counts,
        in_row_offsets,
        in_edge_ids,
        kernel_row_offsets,
        kernel_edge_ids,
    ) = native
    relation_out_coords = (
        native_out_coords if out_coords is None else out_coords
    )
    return KernelRelation(
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets=row_offsets,
        counts=counts,
        in_row_offsets=in_row_offsets,
        in_edge_ids=in_edge_ids,
        kernel_row_offsets=kernel_row_offsets,
        kernel_edge_ids=kernel_edge_ids,
        kernel_offsets=offsets,
        out_coords=relation_out_coords,
        n_in_capacity=in_capacity,
        n_out_capacity=int(relation_out_coords.shape[0]),
        n_kernels=len(offsets),
        source_coords=source_coords,
        source_active_rows=source_active_rows,
        target_coords=target_coords,
        target_active_rows=target_active_rows,
        stride=stride,
        padding=padding,
        kind=kind,
    )


def _neighbor_relation_from_native(
    native: NativeNeighborRelation,
    *,
    query_capacity: int,
    source_capacity: int,
    max_neighbors: int,
) -> NeighborRelation:
    (
        query_rows,
        source_rows,
        neighbor_ids,
        distances,
        row_offsets,
        counts,
    ) = native
    return NeighborRelation(
        query_rows,
        source_rows,
        neighbor_ids,
        distances,
        row_offsets=row_offsets,
        counts=counts,
        n_query_capacity=query_capacity,
        n_source_capacity=source_capacity,
        max_neighbors=max_neighbors,
    )


# MARK: - helpers


def _offset_array(offsets: tuple[Triple, ...]) -> mx.array:
    return mx.array(offsets, dtype=mx.int32)


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')


def _require_nonnegative(values: Triple, name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f'{name} values must be non-negative.')


def _require_matching_coord_dtype(lhs: mx.array, rhs: mx.array) -> None:
    if lhs.dtype != rhs.dtype:
        raise ValueError('coordinate arrays must have matching dtype.')


def _positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f'{name} must be positive.')
    return out


def _nonnegative_float(value: float, name: str) -> float:
    out = float(value)
    if out < 0:
        raise ValueError(f'{name} must be non-negative.')
    return out


def _active_rows(value: mx.array | None, coords: mx.array) -> mx.array:
    if value is not None:
        if value.shape != (1,) or value.dtype != mx.int32:
            raise ValueError(
                'active_rows must have shape (1,) and int32 dtype.'
            )
        return value
    return mx.array([coords.shape[0]], dtype=mx.int32)


def _radius_neighbor_capacity(radius: float) -> int:
    limit = math.ceil(radius)
    radius_squared = radius * radius
    count = 0
    for dz in range(-limit, limit + 1):
        for dy in range(-limit, limit + 1):
            for dx in range(-limit, limit + 1):
                if dx * dx + dy * dy + dz * dz <= radius_squared:
                    count += 1
    return max(count, 1)

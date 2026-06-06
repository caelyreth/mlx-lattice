from __future__ import annotations

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice.core import (  # noqa: E402
    KernelMap,
    KernelSpec,
)


def test_kernel_spec_normalizes_and_classifies_common_paths() -> None:
    pointwise = KernelSpec(size=1)
    subm = KernelSpec(size=(3, 3, 3))

    assert pointwise.size == (1, 1, 1)
    assert pointwise.volume == 1
    assert pointwise.is_pointwise
    assert pointwise.is_centered_submanifold
    assert subm.volume == 27
    assert subm.is_centered_submanifold
    assert not KernelSpec(size=2).is_centered_submanifold


def test_kernel_spec_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match='kernel_size'):
        KernelSpec(size=0)
    with pytest.raises(ValueError, match='stride'):
        KernelSpec(stride=0)
    with pytest.raises(ValueError, match='padding'):
        KernelSpec(padding=-1)
    with pytest.raises(ValueError, match='dilation'):
        KernelSpec(dilation=0)


def test_kernel_map_accepts_edge_contract() -> None:
    in_rows = mx.array([0, 1, 0], dtype=mx.int32)
    out_rows = mx.array([0, 0, 1], dtype=mx.int32)
    kernel_ids = mx.array([0, 1, 0], dtype=mx.int32)
    out_coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )

    mapping = KernelMap(
        in_rows,
        out_rows,
        kernel_ids,
        kernel_offsets=((0, 0, 0), (1, 0, 0)),
        out_coords=out_coords,
        n_in_rows=2,
        n_kernels=2,
    )

    assert mapping.n_edges == 3
    assert mapping.n_out_rows == 2
    assert mapping.n_in_rows == 2
    assert mapping.n_kernels == 2
    assert mapping.kernel_offsets == ((0, 0, 0), (1, 0, 0))


def test_kernel_map_rejects_shape_and_count_mismatches() -> None:
    rows = mx.array([0, 1], dtype=mx.int32)
    short = mx.array([0], dtype=mx.int32)

    with pytest.raises(ValueError, match='same row count'):
        KernelMap(rows, short, rows)

    with pytest.raises(ValueError, match='n_kernels'):
        KernelMap(
            rows,
            rows,
            rows,
            kernel_offsets=((0, 0, 0),),
            n_kernels=2,
        )

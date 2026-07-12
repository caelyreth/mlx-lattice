from __future__ import annotations

import pytest

from lattice_contract.kernel import (
    centered_kernel_offsets,
    indexed_kernel_offsets,
    kernel_positions,
    kernel_row_permutation,
    sparse_kernel_offsets,
)


def test_kernel_positions_use_xyz_axes_with_z_fastest_rows() -> None:
    assert kernel_positions((2, 2, 2)) == (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    )


def test_centered_and_indexed_offsets_share_row_order() -> None:
    assert indexed_kernel_offsets((3, 1, 1)) == (
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    )
    assert centered_kernel_offsets((3, 1, 1)) == (
        (-1, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
    )
    assert sparse_kernel_offsets((3, 2, 1)) == (
        (-1, 0, 0),
        (-1, 1, 0),
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    )


def test_kernel_row_permutation_is_explicit_and_bijective() -> None:
    source = ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1))
    target = kernel_positions((2, 1, 2))
    assert kernel_row_permutation(source, target) == (0, 2, 1, 3)


def test_kernel_row_permutation_rejects_ambiguous_layout() -> None:
    with pytest.raises(ValueError, match='unique'):
        kernel_row_permutation(
            ((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (1, 0, 0))
        )

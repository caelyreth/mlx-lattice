"""Backend-neutral sparse kernel position conventions.

The portable contract names coordinates as ``(x, y, z)``.  Kernel rows always
follow that axis order, with ``z`` varying fastest.  Backends may pack those
rows for execution, but a checkpoint or artifact must never rely on an
undocumented backend enumeration.
"""

from __future__ import annotations

from collections.abc import Sequence

type Triple = tuple[int, int, int]

CANONICAL_CONV3D_WEIGHT_LAYOUT = 'conv3d_o_xyz_i'

__all__ = [
    'CANONICAL_CONV3D_WEIGHT_LAYOUT',
    'centered_kernel_offsets',
    'indexed_kernel_offsets',
    'kernel_positions',
    'kernel_row_permutation',
    'sparse_kernel_offsets',
]


def kernel_positions(size: int | Sequence[int]) -> tuple[Triple, ...]:
    """Return canonical dense-kernel positions in ``x, y, z`` order."""

    x_size, y_size, z_size = _triple(size, name='size')
    return tuple(
        (x, y, z)
        for x in range(x_size)
        for y in range(y_size)
        for z in range(z_size)
    )


def indexed_kernel_offsets(
    size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> tuple[Triple, ...]:
    """Return forward-convolution offsets in canonical row order."""

    dilation_xyz = _triple(dilation, name='dilation')
    return tuple(
        (
            position[0] * dilation_xyz[0],
            position[1] * dilation_xyz[1],
            position[2] * dilation_xyz[2],
        )
        for position in kernel_positions(size)
    )


def centered_kernel_offsets(
    size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> tuple[Triple, ...]:
    """Return centered odd-kernel offsets in canonical row order."""

    size_xyz = _triple(size, name='size')
    if any(item % 2 == 0 for item in size_xyz):
        raise ValueError(
            'centered kernel offsets require odd kernel sizes.'
        )
    dilation_xyz = _triple(dilation, name='dilation')
    center: Triple = (
        size_xyz[0] // 2,
        size_xyz[1] // 2,
        size_xyz[2] // 2,
    )
    return tuple(
        (
            (position[0] - center[0]) * dilation_xyz[0],
            (position[1] - center[1]) * dilation_xyz[1],
            (position[2] - center[2]) * dilation_xyz[2],
        )
        for position in kernel_positions(size_xyz)
    )


def sparse_kernel_offsets(
    size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> tuple[Triple, ...]:
    """Return sparse-convolution offsets in canonical row order.

    Odd axes are centered around zero, while even axes use their indexed
    forward-convolution positions. This matches sparse support construction for
    mixed-shape kernels without introducing a backend-specific row ordering.
    """

    size_xyz = _triple(size, name='size')
    dilation_xyz = _triple(dilation, name='dilation')
    origin: Triple = (
        size_xyz[0] // 2 if size_xyz[0] % 2 else 0,
        size_xyz[1] // 2 if size_xyz[1] % 2 else 0,
        size_xyz[2] // 2 if size_xyz[2] % 2 else 0,
    )
    return tuple(
        (
            (position[0] - origin[0]) * dilation_xyz[0],
            (position[1] - origin[1]) * dilation_xyz[1],
            (position[2] - origin[2]) * dilation_xyz[2],
        )
        for position in kernel_positions(size_xyz)
    )


def kernel_row_permutation(
    source: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Return source-row indices that realize ``target`` offset order.

    This is intended for one-time checkpoint conversion.  Repeated or missing
    positions are rejected because a row permutation must be bijective.
    """

    source_positions = tuple(
        tuple(int(item) for item in row) for row in source
    )
    target_positions = tuple(
        tuple(int(item) for item in row) for row in target
    )
    if len(source_positions) != len(target_positions):
        raise ValueError('kernel position lists must have equal length.')
    if len(set(source_positions)) != len(source_positions):
        raise ValueError('source kernel positions must be unique.')
    if len(set(target_positions)) != len(target_positions):
        raise ValueError('target kernel positions must be unique.')
    source_rows = {
        position: index for index, position in enumerate(source_positions)
    }
    try:
        return tuple(source_rows[position] for position in target_positions)
    except KeyError as exc:
        raise ValueError(
            'kernel position lists do not describe the same set.'
        ) from exc


def _triple(value: int | Sequence[int], *, name: str) -> Triple:
    if isinstance(value, int):
        values = (int(value),) * 3
    else:
        values = tuple(int(item) for item in value)
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f'{name} must contain three positive integers.')
    return values[0], values[1], values[2]

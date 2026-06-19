from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

import mlx.core as mx

_ARTIFACT_PACKAGE = 'mlx_lattice.backends.cuda.artifacts'
_IMPLEMENTED_OPS = frozenset({'lookup_coords'})


def runtime_available() -> bool:
    cuda = getattr(mx, 'cuda', None)
    return bool(
        cuda is not None
        and cuda.is_available()
        and hasattr(mx.fast, 'precompiled_cuda_kernel')
        and _artifact_exists('coords.ptx')
    )


def selected() -> bool:
    return runtime_available() and mx.default_device() == mx.gpu


def info() -> dict[str, Any]:
    return {
        'available': runtime_available(),
        'api': 'mx.fast.precompiled_cuda_kernel',
        'artifact_package': _ARTIFACT_PACKAGE,
        'implemented_ops': tuple(sorted(_IMPLEMENTED_OPS)),
    }


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    if coords.dtype != mx.int32 or queries.dtype != mx.int32:
        raise ValueError('CUDA lookup_coords requires int32 coordinates.')
    if coords.ndim != 2 or queries.ndim != 2:
        raise ValueError('coords and queries must be rank-2 arrays.')
    if coords.shape[1] != 4 or queries.shape[1] != 4:
        raise ValueError('coords and queries must have shape (N, 4).')

    return _run(
        artifact='coords.ptx',
        name='lookup_coords_i32',
        inputs=[coords, queries],
        output_shapes=[(queries.shape[0],)],
        output_dtypes=[mx.int32],
        scalars=[coords.shape[0], queries.shape[0]],
        grid=(queries.shape[0], 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


def _run(
    *,
    artifact: str,
    name: str,
    inputs: list[mx.array],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[mx.Dtype],
    scalars: list[bool | int | float],
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    init_value: float | None = None,
) -> list[mx.array]:
    return mx.fast.precompiled_cuda_kernel(
        name=name,
        compiled_source=_artifact_bytes(artifact),
        inputs=inputs,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        scalars=scalars,
        grid=grid,
        threadgroup=threadgroup,
        init_value=init_value,
        ensure_row_contiguous=True,
    )


def _artifact_exists(name: str) -> bool:
    try:
        return resources.files(_ARTIFACT_PACKAGE).joinpath(name).is_file()
    except ModuleNotFoundError:
        return False


@cache
def _artifact_bytes(name: str) -> bytes:
    return resources.files(_ARTIFACT_PACKAGE).joinpath(name).read_bytes()

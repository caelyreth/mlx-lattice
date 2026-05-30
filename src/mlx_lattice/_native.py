from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx


def _ext():
    import mlx.core  # noqa: F401

    from . import _ext as native

    return native


def version() -> str:
    return str(_ext().version())


def capabilities() -> Mapping[str, bool]:
    return dict(_ext().capabilities())


def conv3d_feats(
    feats: mx.array,
    weight: mx.array,
    maps: mx.array,
    kernels: mx.array,
    out_rows: int,
    *,
    stream: Any | None = None,
) -> mx.array:
    if stream is None:
        return _ext().conv3d_feats(
            feats,
            weight,
            maps,
            kernels,
            out_rows,
        )
    return _ext().conv3d_feats(
        feats,
        weight,
        maps,
        kernels,
        out_rows,
        stream=stream,
    )

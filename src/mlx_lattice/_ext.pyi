from typing import Any

import mlx.core as mx

def version() -> str: ...
def capabilities() -> dict[str, bool]: ...
def conv3d_feats(
    feats: mx.array,
    weight: mx.array,
    maps: mx.array,
    kernels: mx.array,
    out_rows: int,
    *,
    stream: Any | None = None,
) -> mx.array: ...

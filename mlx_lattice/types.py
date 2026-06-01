from __future__ import annotations

from collections.abc import Sequence

Triple = tuple[int, int, int]


def triple(value: int | Sequence[int], *, name: str) -> Triple:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f'{name} must be an int or a 3-tuple.')
    return (int(value[0]), int(value[1]), int(value[2]))

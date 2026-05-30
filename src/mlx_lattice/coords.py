from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx

from mlx_lattice.point import KernelMap, build_kernel_map
from mlx_lattice.types import Triple, triple


@dataclass(frozen=True)
class CoordinateMapKey:
    id: int
    stride: Triple


@dataclass
class CoordinateManager:
    _next_id: int = 0
    _coords: dict[CoordinateMapKey, mx.array] = field(default_factory=dict)
    _maps: dict[tuple[CoordinateMapKey, Triple, Triple], KernelMap] = field(
        default_factory=dict
    )

    def insert(
        self,
        coords: mx.array,
        stride: int | Sequence[int] = 1,
    ) -> CoordinateMapKey:
        key = CoordinateMapKey(self._next_id, triple(stride, name='stride'))
        self._next_id += 1
        self._coords[key] = coords
        return key

    def coords(self, key: CoordinateMapKey) -> mx.array:
        return self._coords[key]

    def kernel_map(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
    ) -> KernelMap:
        kernel = triple(kernel_size, name='kernel_size')
        step = triple(stride, name='stride')
        cache_key = (key, kernel, step)
        if cache_key not in self._maps:
            self._maps[cache_key] = build_kernel_map(
                self.coords(key), kernel_size=kernel, stride=step
            )
        return self._maps[cache_key]

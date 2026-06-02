from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx

from mlx_lattice.core.coords.builders import (
    build_generative_map,
    build_kernel_map,
    build_transposed_kernel_map,
)
from mlx_lattice.core.coords.set_ops import inverse_map
from mlx_lattice.core.coords.validation import validate_coords
from mlx_lattice.core.maps import KernelMap, KernelSpec
from mlx_lattice.core.types import Triple, triple


@dataclass(frozen=True, slots=True)
class CoordinateMapKey:
    id: int
    stride: Triple


@dataclass(slots=True)
class CoordinateManager:
    _next_id: int = 0
    _coords: dict[CoordinateMapKey, mx.array] = field(default_factory=dict)
    _coord_keys: dict[tuple[int, Triple], CoordinateMapKey] = field(
        default_factory=dict
    )
    _kernel_maps: dict[
        tuple[CoordinateMapKey, KernelSpec, str], KernelMap
    ] = field(default_factory=dict)

    def insert(
        self,
        coords: mx.array,
        stride: int | Sequence[int] = 1,
    ) -> CoordinateMapKey:
        validate_coords(coords)
        normalized = triple(stride, name='stride')
        cache_key = (id(coords), normalized)
        if cache_key in self._coord_keys:
            return self._coord_keys[cache_key]

        key = CoordinateMapKey(self._next_id, normalized)
        self._next_id += 1
        self._coords[key] = coords
        self._coord_keys[cache_key] = key
        return key

    def coords(self, key: CoordinateMapKey) -> mx.array:
        return self._coords[key]

    def inverse_map(
        self,
        source: CoordinateMapKey,
        target: CoordinateMapKey,
    ) -> mx.array:
        return inverse_map(self.coords(source), self.coords(target))

    def kernel_map(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> KernelMap:
        spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        cache_key = (key, spec, 'forward')
        if cache_key not in self._kernel_maps:
            self._kernel_maps[cache_key] = build_kernel_map(
                self.coords(key),
                kernel_size=spec.size,
                stride=spec.stride,
                padding=spec.padding,
                dilation=spec.dilation,
            )
        return self._kernel_maps[cache_key]

    def generative_map(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
    ) -> KernelMap:
        spec = KernelSpec(size=kernel_size, stride=stride)
        cache_key = (key, spec, 'generative')
        if cache_key not in self._kernel_maps:
            self._kernel_maps[cache_key] = build_generative_map(
                self.coords(key),
                kernel_size=spec.size,
                stride=spec.stride,
            )
        return self._kernel_maps[cache_key]

    def transposed_kernel_map(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> KernelMap:
        spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        cache_key = (key, spec, 'transpose')
        if cache_key not in self._kernel_maps:
            self._kernel_maps[cache_key] = build_transposed_kernel_map(
                self.coords(key),
                kernel_size=spec.size,
                stride=spec.stride,
                padding=spec.padding,
                dilation=spec.dilation,
            )
        return self._kernel_maps[cache_key]

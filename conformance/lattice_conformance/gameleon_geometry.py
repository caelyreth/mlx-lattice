"""MLX replay of Gameleon's fixed Level-8 geometry BPP profile.

This module intentionally lives in conformance tooling rather than the public
``mlx_lattice`` API.  Gameleon's FOG and FCG names describe one application;
the reusable runtime primitives are occupancy downsampling, occupancy
expansion, Morton ordering, sparse convolution, and dense feature operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import mlx.core as mx

from mlx_lattice import SparseTensor
from mlx_lattice.core.coords import SparseOccupancy
from mlx_lattice.ops import (
    linear_features,
    morton_sort_coords,
    occupancy_downsample,
    occupancy_expand,
    relu,
    sparse_add,
    subm_conv3d,
)

_KERNEL_SIZE = 3


@dataclass(frozen=True, slots=True)
class GeometryProfileWeights:
    """Portable trained tensors for the fixed 32-channel geometry profile."""

    values: Mapping[str, mx.array]

    @classmethod
    def from_safetensors(cls, path: str | Path) -> GeometryProfileWeights:
        """Load a portable safetensors state mapping.

        Sparse kernels use canonical lattice relation order in ``(K, Cin,
        Cout)`` layout. Legacy TorchSparse checkpoints must be converted by
        the CUDA-side conformance tool before MLX replay.
        """

        loaded = mx.load(str(path))
        if not isinstance(loaded, dict):
            raise ValueError('geometry weights must be a tensor mapping.')
        values = cast('dict[str, mx.array]', loaded)
        legacy = sorted(key for key in values if key.endswith('.kernel'))
        if legacy:
            raise ValueError(
                'geometry weights contain legacy TorchSparse .kernel keys; '
                'run the CUDA-side convert-checkpoint tool first: '
                + ', '.join(legacy)
            )
        return cls(values)

    def tensor(self, name: str, *, dtype: mx.Dtype) -> mx.array:
        """Return a named floating tensor in the profile feature dtype."""

        try:
            value = self.values[name]
        except KeyError as exc:
            raise ValueError(
                f'geometry profile is missing weight: {name}'
            ) from exc
        if value.dtype not in (mx.float16, mx.float32):
            raise ValueError(
                f'geometry weight {name} must be floating point.'
            )
        return value if value.dtype == dtype else value.astype(dtype)


@dataclass(frozen=True, slots=True)
class GeometryProfileStage:
    """One teacher-forced hierarchy prediction stage."""

    depth: int
    parent_points: int
    target_points: int
    group1_points: int
    group2_points: int


@dataclass(frozen=True, slots=True)
class GeometryProfileResult:
    """Forward-BPP output and structural diagnostics for one support cloud."""

    bits: mx.array
    bpp: mx.array
    input_points: int
    levels: tuple[int, ...]
    stages: tuple[GeometryProfileStage, ...]


@dataclass(frozen=True, slots=True)
class GeometryProfileTiming:
    """Warm runtime timing that excludes input parsing and tensor creation."""

    bpp: float
    samples_ms: tuple[float, ...]
    result: GeometryProfileResult

    @property
    def median_ms(self) -> float:
        ordered = sorted(self.samples_ms)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2


class Level8GeometryProfile:
    """Replay the active four-stage Gameleon Level-8 BPP network on MLX.

    The source model stops FOG when fewer than 64 parents remain and applies
    teacher-forced occupancy conditioning at every resulting pyramid stage.
    It estimates neural BPP only; arithmetic coding and file I/O remain codec
    policy outside this profile.
    """

    def __init__(
        self,
        weights: GeometryProfileWeights,
        *,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        if dtype not in (mx.float16, mx.float32):
            raise ValueError(
                'geometry profile dtype must be float16 or float32.'
            )
        self.weights = weights
        self.dtype = dtype

    def from_xyz(self, xyz: mx.array) -> SparseTensor:
        """Create the single-batch unit-feature support used by the codec."""

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError('xyz must have shape (N, 3).')
        if xyz.dtype not in (mx.int32, mx.int64):
            raise ValueError('xyz must use an integer dtype.')
        spatial = xyz if xyz.dtype == mx.int32 else xyz.astype(mx.int32)
        batch = mx.zeros((spatial.shape[0], 1), dtype=mx.int32)
        coords = mx.concatenate([batch, spatial], axis=1)
        features = mx.ones((spatial.shape[0], 1), dtype=self.dtype)
        return SparseTensor(coords, features)

    def build_pyramid(self, x: SparseTensor) -> tuple[SparseOccupancy, ...]:
        """Build and compact the FOG occupancy hierarchy.

        The generic coordinate primitive uses static-capacity buffers.  The
        source network has dynamic sparse shapes, so compacting each level here
        preserves its actual support and avoids carrying inactive capacity into
        the sparse convolution stages.
        """

        levels: list[SparseOccupancy] = []
        coords = _compact_sparse_coords(x)
        active = mx.array([coords.shape[0]], dtype=mx.int32)
        while True:
            level = _compact_occupancy(occupancy_downsample(coords, active))
            levels.append(level)
            if level.coords.shape[0] < 64:
                return tuple(reversed(levels))
            coords = level.coords
            active = level.active_rows

    def __call__(self, x: SparseTensor) -> GeometryProfileResult:
        """Run forward BPP from an already constructed sparse support tensor."""

        if x.channels != 1:
            raise ValueError(
                'geometry profile expects one input feature channel.'
            )
        if x.feats.dtype != self.dtype:
            x = x.astype(self.dtype)
        input_points = _active_count(x.active_rows)
        if input_points == 0:
            raise ValueError(
                'geometry profile requires at least one point.'
            )

        levels = self.build_pyramid(x)
        group1_bits = mx.array(0.0, dtype=self.dtype)
        group2_bits = mx.array(0.0, dtype=self.dtype)
        stages: list[GeometryProfileStage] = []

        for depth, (parent, target) in enumerate(pairwise(levels)):
            target_coords, target_occupancy = _sort_level(target)
            candidates = self._predict_candidates(parent)
            _require_same_coords(candidates.coords, target_coords, depth)
            group1, group2 = _checkerboard_masks(target_coords)
            group1_count = _mask_count(group1)
            group2_count = _mask_count(group2)
            group1_rows = _selected_rows(group1, group1_count)
            group2_rows = _selected_rows(group2, group2_count)

            if group1_count:
                features = mx.take(candidates.feats, group1_rows, axis=0)
                occupancy = mx.take(target_occupancy, group1_rows, axis=0)
                bits = self._group_bits(
                    coords=mx.take(target_coords, group1_rows, axis=0),
                    features=features,
                    occupancy=occupancy,
                    context='group1',
                )
                group1_bits = group1_bits + bits

            if group2_count:
                features = mx.take(candidates.feats, group2_rows, axis=0)
                if group1_count:
                    features = self._group2_features(
                        target_coords,
                        candidates.feats,
                        group1,
                        group2,
                        group2_rows,
                        target_occupancy,
                    )
                occupancy = mx.take(target_occupancy, group2_rows, axis=0)
                bits = self._group_bits(
                    coords=mx.take(target_coords, group2_rows, axis=0),
                    features=features,
                    occupancy=occupancy,
                    context='group2',
                )
                group2_bits = group2_bits + bits

            stages.append(
                GeometryProfileStage(
                    depth=depth,
                    parent_points=parent.coords.shape[0],
                    target_points=target.coords.shape[0],
                    group1_points=group1_count,
                    group2_points=group2_count,
                )
            )

        bits = group1_bits + group2_bits
        return GeometryProfileResult(
            bits=bits,
            bpp=bits / mx.array(input_points, dtype=self.dtype),
            input_points=input_points,
            levels=tuple(level.coords.shape[0] for level in levels),
            stages=tuple(stages),
        )

    def benchmark(
        self,
        x: SparseTensor,
        *,
        warmup: int = 2,
        repeats: int = 10,
    ) -> GeometryProfileTiming:
        """Time warm profile calls with construction of ``x`` excluded."""

        if warmup < 0 or repeats < 1:
            raise ValueError(
                'warmup must be non-negative and repeats positive.'
            )
        for _ in range(warmup):
            result = self(x)
            mx.eval(result.bpp)

        samples = []
        result = self(x)
        for _ in range(repeats):
            start = perf_counter()
            result = self(x)
            mx.eval(result.bpp)
            samples.append((perf_counter() - start) * 1_000.0)
        return GeometryProfileTiming(
            bpp=float(result.bpp.item()),
            samples_ms=tuple(samples),
            result=result,
        )

    def _predict_candidates(self, parent: SparseOccupancy) -> SparseTensor:
        parent_features = self._embedding(
            'prior_embedding.weight', parent.occupancy
        )
        prior = self._resnet(
            SparseTensor(parent.coords, parent_features), 'prior_resnet'
        )
        expanded = occupancy_expand(
            parent.coords,
            parent.occupancy,
            parent.active_rows,
        )
        child_coords, parent_rows = _compact_expansion(expanded)
        child_features = mx.take(prior.feats, parent_rows, axis=0)
        child_coords, child_features = _sort_coords_and_features(
            child_coords, child_features
        )
        child_features = child_features + self._target_embedding(
            child_coords
        )
        return self._resnet(
            SparseTensor(child_coords, child_features), 'target_resnet'
        )

    def _resnet(self, x: SparseTensor, prefix: str) -> SparseTensor:
        x = relu(self._conv(x, f'{prefix}.0'))
        x = self._residual(x, f'{prefix}.2')
        return self._residual(x, f'{prefix}.3')

    def _residual(self, x: SparseTensor, prefix: str) -> SparseTensor:
        out = relu(self._conv(x, f'{prefix}.conv0'))
        out = self._conv(out, f'{prefix}.conv1')
        return relu(sparse_add(out, x, join='inner'))

    def _spatial_context(
        self, x: SparseTensor, prefix: str
    ) -> SparseTensor:
        return self._conv(relu(self._conv(x, f'{prefix}.0')), f'{prefix}.2')

    def _group_bits(
        self,
        *,
        coords: mx.array,
        features: mx.array,
        occupancy: mx.array,
        context: str,
    ) -> mx.array:
        symbols0 = occupancy % 16
        symbols1 = occupancy // 16
        sparse = SparseTensor(coords, features)
        first = self._spatial_context(sparse, f'{context}_spatial_conv_s0')
        probability0 = self._prediction_head(
            f'{context}_pred_head_s0', first.feats
        )
        selected0 = _select_probability(probability0, symbols0)

        conditioned = features + self._embedding(
            f'{context}_pred_head_s1_emb.weight', symbols0
        )
        second = self._spatial_context(
            sparse.replace(feats=conditioned),
            f'{context}_spatial_conv_s1',
        )
        probability1 = self._prediction_head(
            f'{context}_pred_head_s1', second.feats
        )
        selected1 = _select_probability(probability1, symbols1)
        return _bits(selected0) + _bits(selected1)

    def _group2_features(
        self,
        coords: mx.array,
        features: mx.array,
        group1: mx.array,
        group2: mx.array,
        group2_rows: mx.array,
        occupancy: mx.array,
    ) -> mx.array:
        group1_embedding = self._embedding(
            'prior_embedding.weight', occupancy
        )
        visible = mx.where(group2[:, None], 0, features)
        visible = visible + mx.where(group1[:, None], group1_embedding, 0)
        aggregated = self._conv(
            SparseTensor(coords, visible), 'neighbor_conv'
        )
        combined = mx.concatenate(
            [
                mx.take(features, group2_rows, axis=0),
                mx.take(aggregated.feats, group2_rows, axis=0),
            ],
            axis=1,
        )
        fused = linear_features(
            combined,
            self.weights.tensor(
                'feature_fusion.0.weight', dtype=self.dtype
            ),
            self.weights.tensor('feature_fusion.0.bias', dtype=self.dtype),
        )
        return mx.maximum(fused, 0)

    def _prediction_head(self, prefix: str, features: mx.array) -> mx.array:
        out = features
        for index in (0, 2):
            out = linear_features(
                out,
                self.weights.tensor(
                    f'{prefix}.{index}.weight', dtype=self.dtype
                ),
                self.weights.tensor(
                    f'{prefix}.{index}.bias', dtype=self.dtype
                ),
            )
            out = mx.maximum(out, 0)
        out = linear_features(
            out,
            self.weights.tensor(f'{prefix}.4.weight', dtype=self.dtype),
            self.weights.tensor(f'{prefix}.4.bias', dtype=self.dtype),
        )
        return mx.softmax(out, axis=-1)

    def _target_embedding(self, coords: mx.array) -> mx.array:
        delta = coords[:, 1:] % 2
        indices = delta[:, 0] + delta[:, 1] * 2 + delta[:, 2] * 4
        return self._embedding(
            'target_embedding.target_res_embedding.weight', indices
        )

    def _embedding(self, name: str, indices: mx.array) -> mx.array:
        table = self.weights.tensor(name, dtype=self.dtype)
        return mx.take(table, indices.astype(mx.int32), axis=0)

    def _conv(self, x: SparseTensor, prefix: str) -> SparseTensor:
        return subm_conv3d(
            x,
            self.weights.tensor(f'{prefix}.weight', dtype=self.dtype),
            kernel_size=_KERNEL_SIZE,
        )


def read_ascii_ply_xyz(path: str | Path) -> mx.array:
    """Read integer xyz coordinates from the release's ASCII PLY format."""

    lines = Path(path).read_text(encoding='ascii').splitlines()
    header_end, vertex_count, properties = _ply_header(lines, path)
    if 'format ascii 1.0' not in lines[:header_end]:
        raise ValueError(
            'geometry profile currently accepts ASCII PLY only.'
        )
    try:
        x_index, y_index, z_index = (
            properties.index('x'),
            properties.index('y'),
            properties.index('z'),
        )
    except ValueError as exc:
        raise ValueError(
            'PLY vertex data must contain x, y, and z.'
        ) from exc
    values = []
    for line in lines[header_end + 1 : header_end + 1 + vertex_count]:
        if not line.strip():
            continue
        fields = line.split()
        values.append(
            [
                round(float(fields[x_index])),
                round(float(fields[y_index])),
                round(float(fields[z_index])),
            ]
        )
    if len(values) != vertex_count:
        raise ValueError(
            f'PLY declares {vertex_count} vertices but contains {len(values)}.'
        )
    return mx.array(values, dtype=mx.int32)


def _ply_header(
    lines: list[str], path: str | Path
) -> tuple[int, int, list[str]]:
    if not lines or lines[0] != 'ply':
        raise ValueError(f'not a PLY file: {path}')
    vertex_count: int | None = None
    properties: list[str] = []
    current_element = ''
    for index, line in enumerate(lines[1:], start=1):
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'element' and len(fields) == 3:
            current_element = fields[1]
            if current_element == 'vertex':
                vertex_count = int(fields[2])
        elif (
            fields[0] == 'property'
            and current_element == 'vertex'
            and len(fields) == 3
        ):
            properties.append(fields[2])
        elif fields[0] == 'end_header':
            if vertex_count is None:
                raise ValueError(f'PLY has no vertex element: {path}')
            return index, vertex_count, properties
    raise ValueError(f'PLY header is missing end_header: {path}')


def _compact_sparse_coords(x: SparseTensor) -> mx.array:
    return x.coords[: _active_count(x.active_rows)]


def _compact_occupancy(value: SparseOccupancy) -> SparseOccupancy:
    active = _active_count(value.active_rows)
    return SparseOccupancy(
        value.coords[:active],
        mx.array([active], dtype=mx.int32),
        value.occupancy[:active],
    )


def _compact_expansion(value: Any) -> tuple[mx.array, mx.array]:
    active = _active_count(value.active_rows)
    return value.coords[:active], value.parent_rows[:active]


def _sort_level(value: SparseOccupancy) -> tuple[mx.array, mx.array]:
    return _sort_coords_and_features(value.coords, value.occupancy)


def _sort_coords_and_features(
    coords: mx.array, features: mx.array
) -> tuple[mx.array, mx.array]:
    ordering = morton_sort_coords(coords)
    return ordering.coords, mx.take(features, ordering.order, axis=0)


def _checkerboard_masks(coords: mx.array) -> tuple[mx.array, mx.array]:
    group1 = mx.equal(mx.sum(coords[:, 1:], axis=1) % 2, 0)
    return group1, mx.logical_not(group1)


def _select_probability(
    probabilities: mx.array, symbols: mx.array
) -> mx.array:
    rows = symbols.astype(mx.int32).reshape(-1, 1)
    return mx.take_along_axis(probabilities, rows, axis=1)


def _bits(probabilities: mx.array) -> mx.array:
    return mx.sum(mx.clip(-mx.log2(probabilities + 1e-10), 0, 50))


def _active_count(active_rows: mx.array) -> int:
    return int(active_rows.item())


def _mask_count(mask: mx.array) -> int:
    return int(mx.sum(mask).item())


def _selected_rows(mask: mx.array, count: int) -> mx.array:
    rows = mx.arange(mask.shape[0], dtype=mx.int32)
    ordering = mx.argsort(mx.where(mask, rows, rows + mask.shape[0]))
    return ordering[:count].astype(mx.int32)


def _require_same_coords(
    actual: mx.array, expected: mx.array, depth: int
) -> None:
    if not mx.array_equal(actual, expected).item():
        raise ValueError(
            f'candidate coordinates differ from target support at depth {depth}.'
        )


__all__ = [
    'GeometryProfileResult',
    'GeometryProfileStage',
    'GeometryProfileTiming',
    'GeometryProfileWeights',
    'Level8GeometryProfile',
    'read_ascii_ply_xyz',
]

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from lattice_conformance.archive import fixture_root
from lattice_conformance.metrics import distribution
from mlx_lattice import SparseTensor
from mlx_lattice.artifact import (
    load_lattice_program,
    native_artifact_execution_available,
)
from mlx_lattice.core.coords import morton_sort_coords


@dataclass(frozen=True, slots=True)
class CaseStats:
    name: str
    family: str
    output_kind: str
    max_abs: float
    max_rel: float


@dataclass(frozen=True, slots=True)
class ReplayReport:
    fixture_root: Path
    schema: str | None
    stats: list[CaseStats]
    failures: list[str]

    @property
    def summary(self) -> dict[str, Any]:
        abs_values = [item.max_abs for item in self.stats]
        rel_values = [item.max_rel for item in self.stats]
        return {
            'case_count': len(self.stats),
            'failed': len(self.failures),
            'abs': distribution(abs_values),
            'rel': distribution(rel_values),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            'fixture_root': str(self.fixture_root),
            'schema': self.schema,
            'case_count': len(self.stats),
            'summary': self.summary,
            'failures': self.failures,
            'cases': [
                {
                    'name': item.name,
                    'family': item.family,
                    'output_kind': item.output_kind,
                    'max_abs': item.max_abs,
                    'max_rel': item.max_rel,
                }
                for item in self.stats
            ],
        }


def replay_fixtures(path: Path, *, fail_fast: bool = False) -> ReplayReport:
    if not native_artifact_execution_available():
        raise RuntimeError(
            'fixture replay requires an MLIR-enabled native extension.'
        )
    failures: list[str] = []
    stats: list[CaseStats] = []
    with fixture_root(path) as root:
        manifest = json.loads((root / 'manifest.json').read_text())
        for item in manifest['cases']:
            case = root / item['name']
            try:
                stats.append(_check_case(case, item))
            except Exception as exc:
                failures.append(
                    f'{item["name"]}: {type(exc).__name__}: {exc}'
                )
                if fail_fast:
                    break
    return ReplayReport(
        fixture_root=path,
        schema=manifest.get('schema'),
        stats=stats,
        failures=failures,
    )


def _check_case(case: Path, metadata: dict[str, Any]) -> CaseStats:
    program = load_lattice_program(case)
    inputs = cast(
        'dict[str, mx.array]', mx.load(str(case / 'inputs.safetensors'))
    )
    expected = cast(
        'dict[str, mx.array]', mx.load(str(case / 'expected.safetensors'))
    )
    output = program(**inputs)
    rtol = float(metadata.get('rtol', 2e-3))
    atol = float(metadata.get('atol', 2e-3))
    if metadata['output_kind'] == 'sparse':
        max_abs, max_rel = _assert_sparse_close(
            output, expected, rtol=rtol, atol=atol
        )
        return CaseStats(
            name=str(metadata['name']),
            family=str(metadata['family']),
            output_kind='sparse',
            max_abs=max_abs,
            max_rel=max_rel,
        )
    dense_output = cast('mx.array', output)
    expected_dense = expected['output']
    mx.eval(dense_output, expected_dense)
    max_abs, max_rel = _error_stats(dense_output, expected_dense)
    if not mx.allclose(
        dense_output, expected_dense, rtol=rtol, atol=atol
    ).item():
        raise AssertionError('dense output mismatch')
    return CaseStats(
        name=str(metadata['name']),
        family=str(metadata['family']),
        output_kind='dense',
        max_abs=max_abs,
        max_rel=max_rel,
    )


def _assert_sparse_close(
    output: object,
    expected: dict[str, mx.array],
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    if not isinstance(output, SparseTensor):
        raise TypeError(
            f'expected SparseTensor output, got {type(output)!r}'
        )
    mx.eval(output.coords, output.feats, expected['output.features'])
    if not mx.array_equal(
        output.active_rows, expected['output.active']
    ).item():
        raise AssertionError('sparse active row mismatch')
    active = int(output.active_rows.item())
    actual_coords, actual_features = _canonical_sparse_rows(
        output.coords[:active], output.feats[:active]
    )
    expected_coords, expected_features = _canonical_sparse_rows(
        expected['output.coords'], expected['output.features']
    )
    if not mx.array_equal(actual_coords, expected_coords).item():
        raise AssertionError('sparse coordinate mismatch')
    max_abs, max_rel = _error_stats(actual_features, expected_features)
    if not mx.allclose(
        actual_features,
        expected_features,
        rtol=rtol,
        atol=atol,
    ).item():
        raise AssertionError('sparse feature mismatch')
    return max_abs, max_rel


def _canonical_sparse_rows(
    coords: mx.array, features: mx.array
) -> tuple[mx.array, mx.array]:
    """Align sparse rows by coordinate value before cross-runtime comparison."""

    ordering = morton_sort_coords(coords)
    return ordering.coords, mx.take(features, ordering.order, axis=0)


def _error_stats(
    actual: mx.array, expected: mx.array
) -> tuple[float, float]:
    diff = mx.abs(actual - expected)
    denom = mx.maximum(mx.abs(actual), mx.abs(expected))
    rel = diff / mx.maximum(denom, mx.array(1e-12, dtype=denom.dtype))
    mx.eval(diff, rel)
    return float(mx.max(diff).item()), float(mx.max(rel).item())


__all__ = ['CaseStats', 'ReplayReport', 'replay_fixtures']

from __future__ import annotations

import argparse
import json
import statistics
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from mlx_lattice import SparseTensor
from mlx_lattice.artifact import (
    load_lattice_program,
    native_artifact_execution_available,
)


@dataclass(frozen=True, slots=True)
class CaseStats:
    name: str
    family: str
    output_kind: str
    max_abs: float
    max_rel: float


def main() -> None:
    args = _parse_args()
    if not native_artifact_execution_available():
        raise SystemExit(
            'fuzz fixture replay requires an MLIR-enabled native extension.'
        )
    root = _fixture_root(Path(args.path))
    failures: list[str] = []
    stats: list[CaseStats] = []
    manifest = json.loads((root / 'manifest.json').read_text())
    for item in manifest['cases']:
        case = root / item['name']
        try:
            stats.append(_check_case(case, item))
        except Exception as exc:
            failures.append(f'{item["name"]}: {type(exc).__name__}: {exc}')
            if args.fail_fast:
                break
    if failures:
        for failure in failures:
            print(f'FAIL {failure}')
        raise SystemExit(1)
    report = _report(root, manifest, stats)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary'], indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Replay Torch CUDA generated lattice fuzz fixtures on MLX.'
    )
    parser.add_argument(
        'path', help='Fixture directory or .tar.gz archive.'
    )
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument(
        '--report',
        help='Optional JSON report path for per-case replay accuracy metrics.',
    )
    return parser.parse_args()


def _fixture_root(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.suffixes[-2:] != ['.tar', '.gz']:
        raise ValueError(
            f'expected fixture directory or .tar.gz archive: {path}'
        )
    temp = Path(tempfile.mkdtemp(prefix='mlx_lattice_fuzz_'))
    with tarfile.open(path, 'r:gz') as archive:
        archive.extractall(temp, filter='data')
    roots = [item for item in temp.iterdir() if item.is_dir()]
    if len(roots) != 1:
        raise ValueError(f'archive must contain one root directory: {path}')
    return roots[0]


def _check_case(case: Path, metadata: dict[str, Any]) -> CaseStats:
    program = load_lattice_program(case)
    inputs = cast(
        dict[str, mx.array], mx.load(str(case / 'inputs.safetensors'))
    )
    expected = cast(
        dict[str, mx.array], mx.load(str(case / 'expected.safetensors'))
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
    dense_output = cast(mx.array, output)
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
    if not mx.array_equal(
        output.coords[:active], expected['output.coords']
    ).item():
        raise AssertionError('sparse coordinate mismatch')
    features = output.feats[:active]
    max_abs, max_rel = _error_stats(features, expected['output.features'])
    if not mx.allclose(
        features,
        expected['output.features'],
        rtol=rtol,
        atol=atol,
    ).item():
        raise AssertionError('sparse feature mismatch')
    return max_abs, max_rel


def _error_stats(
    actual: mx.array, expected: mx.array
) -> tuple[float, float]:
    diff = mx.abs(actual - expected)
    denom = mx.maximum(mx.abs(actual), mx.abs(expected))
    rel = diff / mx.maximum(denom, mx.array(1e-12, dtype=denom.dtype))
    mx.eval(diff, rel)
    return float(mx.max(diff).item()), float(mx.max(rel).item())


def _report(
    root: Path,
    manifest: dict[str, Any],
    stats: list[CaseStats],
) -> dict[str, Any]:
    abs_values = [item.max_abs for item in stats]
    rel_values = [item.max_rel for item in stats]
    return {
        'fixture_root': str(root),
        'schema': manifest.get('schema'),
        'case_count': len(stats),
        'summary': {
            'case_count': len(stats),
            'abs': _distribution(abs_values),
            'rel': _distribution(rel_values),
        },
        'cases': [
            {
                'name': item.name,
                'family': item.family,
                'output_kind': item.output_kind,
                'max_abs': item.max_abs,
                'max_rel': item.max_rel,
            }
            for item in stats
        ],
    }


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            'avg': 0.0,
            'median': 0.0,
            'p95': 0.0,
            'p99': 0.0,
            'max': 0.0,
        }
    ordered = sorted(values)
    return {
        'avg': statistics.fmean(ordered),
        'median': statistics.median(ordered),
        'p95': _percentile(ordered, 0.95),
        'p99': _percentile(ordered, 0.99),
        'max': ordered[-1],
    }


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


if __name__ == '__main__':
    main()

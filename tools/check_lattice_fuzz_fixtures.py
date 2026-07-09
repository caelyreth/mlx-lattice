from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from mlx_lattice import SparseTensor
from mlx_lattice.artifact import (
    load_lattice_program,
    native_artifact_execution_available,
)


def main() -> None:
    args = _parse_args()
    if not native_artifact_execution_available():
        raise SystemExit(
            'fuzz fixture replay requires an MLIR-enabled native extension.'
        )
    root = _fixture_root(Path(args.path))
    failures: list[str] = []
    manifest = json.loads((root / 'manifest.json').read_text())
    for item in manifest['cases']:
        case = root / item['name']
        try:
            _check_case(case, item)
        except Exception as exc:
            failures.append(f'{item["name"]}: {type(exc).__name__}: {exc}')
            if args.fail_fast:
                break
    if failures:
        for failure in failures:
            print(f'FAIL {failure}')
        raise SystemExit(1)
    print(
        f'checked {len(manifest["cases"])} fuzz fixture cases from {root}'
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Replay Torch CUDA generated lattice fuzz fixtures on MLX.'
    )
    parser.add_argument(
        'path', help='Fixture directory or .tar.gz archive.'
    )
    parser.add_argument('--fail-fast', action='store_true')
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


def _check_case(case: Path, metadata: dict[str, Any]) -> None:
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
        _assert_sparse_close(output, expected, rtol=rtol, atol=atol)
        return
    dense_output = cast(mx.array, output)
    expected_dense = expected['output']
    mx.eval(dense_output, expected_dense)
    if not mx.allclose(
        dense_output, expected_dense, rtol=rtol, atol=atol
    ).item():
        raise AssertionError('dense output mismatch')


def _assert_sparse_close(
    output: object,
    expected: dict[str, mx.array],
    *,
    rtol: float,
    atol: float,
) -> None:
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
    if not mx.allclose(
        output.feats[:active],
        expected['output.features'],
        rtol=rtol,
        atol=atol,
    ).item():
        raise AssertionError('sparse feature mismatch')


if __name__ == '__main__':
    main()

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice import _ext as ext
from mlx_lattice.artifact import load_lattice_program
from tests.support import mx

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / 'fixtures' / 'torch_lattice_e2e'
)


pytestmark = pytest.mark.skipif(
    not hasattr(ext, 'lattice_mlir_plan'),
    reason='Torch lattice E2E fixtures require MLIR-enabled native extension.',
)


def test_torch_lattice_sparse_classifier_artifact_runs_on_mlx() -> None:
    case = FIXTURE_ROOT / 'sparse_classifier'
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))['output']

    output = program(_sparse_input(inputs, batch_counts=(3, 2)))

    mx.eval(output, expected)
    assert mx.allclose(output, expected, rtol=5e-4, atol=5e-4).item()


@pytest.mark.parametrize(
    ('case_name', 'rtol', 'atol'),
    [
        ('quantized_classifier_int8', 1e-2, 1e-2),
        ('quantized_classifier_int4', 5e-2, 5e-2),
    ],
)
def test_torch_lattice_quantized_classifier_artifact_runs_on_mlx(
    case_name: str,
    rtol: float,
    atol: float,
) -> None:
    case = FIXTURE_ROOT / case_name
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))['output']

    output = program(_sparse_input(inputs, batch_counts=(3, 2)))

    mx.eval(output, expected)
    assert mx.allclose(output, expected, rtol=rtol, atol=atol).item()


def test_torch_lattice_target_branch_artifact_runs_on_mlx() -> None:
    case = FIXTURE_ROOT / 'target_branch'
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))

    output = program(**inputs)

    assert isinstance(output, SparseTensor)
    mx.eval(output.coords, output.feats, expected['output.features'])
    assert mx.array_equal(
        output.active_rows, expected['output.active']
    ).item()
    active = int(output.active_rows.item())
    assert mx.array_equal(
        output.coords[:active],
        expected['output.coords'],
    ).item()
    assert mx.allclose(
        output.feats[:active],
        expected['output.features'],
        rtol=2e-3,
        atol=2e-3,
    ).item()


def test_torch_lattice_point_voxel_artifact_runs_on_mlx() -> None:
    case = FIXTURE_ROOT / 'point_voxel'
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))['output']

    output = program(**inputs)

    mx.eval(output, expected)
    assert mx.allclose(output, expected, rtol=1e-4, atol=1e-4).item()


@pytest.mark.parametrize(
    ('case_name', 'batch_counts'),
    [
        ('transpose_convolution', (4,)),
        ('generative_transpose_convolution', (2,)),
    ],
)
def test_torch_lattice_transpose_artifact_runs_on_mlx(
    case_name: str,
    batch_counts: tuple[int, ...],
) -> None:
    case = FIXTURE_ROOT / case_name
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))

    output = program(_sparse_input(inputs, batch_counts=batch_counts))

    _assert_sparse_output_close(output, expected, rtol=2e-3, atol=2e-3)


def _sparse_input(
    inputs: dict[str, mx.array],
    *,
    batch_counts: tuple[int, ...],
) -> SparseTensor:
    return SparseTensor(
        inputs['coords'],
        inputs['features'],
        active_rows=inputs['active'],
        batch_counts=batch_counts,
    )


def _assert_sparse_output_close(
    output: object,
    expected: dict[str, mx.array],
    *,
    rtol: float,
    atol: float,
) -> None:
    assert isinstance(output, SparseTensor)
    mx.eval(output.coords, output.feats, expected['output.features'])
    assert mx.array_equal(
        output.active_rows, expected['output.active']
    ).item()
    active = int(output.active_rows.item())
    assert mx.array_equal(
        output.coords[:active],
        expected['output.coords'],
    ).item()
    assert mx.allclose(
        output.feats[:active],
        expected['output.features'],
        rtol=rtol,
        atol=atol,
    ).item()

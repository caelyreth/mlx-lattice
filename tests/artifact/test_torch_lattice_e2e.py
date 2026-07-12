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

    output = program(
        _sparse_input(inputs, prefix='x_', batch_counts=(3, 2))
    )

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

    output = program(
        _sparse_input(inputs, prefix='x_', batch_counts=(3, 2))
    )

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


def test_torch_lattice_pool_transpose_artifact_runs_on_mlx() -> None:
    case = FIXTURE_ROOT / 'pool_transpose'
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))

    output = program(
        source=_sparse_input(
            inputs,
            prefix='source_',
            batch_counts=(2,),
            stride=(2, 1, 1),
        ),
        target=_sparse_input(
            inputs,
            prefix='target_',
            batch_counts=(4,),
        ),
    )

    _assert_sparse_output_close(output, expected, rtol=1e-5, atol=1e-6)


def test_gameleon_reproduction_block_artifact_runs_on_mlx() -> None:
    case = FIXTURE_ROOT / 'gameleon_reproduction_block'
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))

    output = program(_sparse_input(inputs, prefix='x_', batch_counts=(8,)))

    _assert_sparse_output_close(output, expected, rtol=3e-3, atol=3e-3)


def test_torch_lattice_canonical_kernel_layout_artifact_runs_on_mlx() -> (
    None
):
    """Exercise an exported non-cubic convolution with row-distinct weights."""

    case = FIXTURE_ROOT / 'canonical_kernel_layout'
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))

    output = program(_sparse_input(inputs, prefix='x_', batch_counts=(15,)))

    _assert_sparse_output_close(output, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ('case_name', 'source_stride', 'target_rows', 'rtol', 'atol'),
    [
        ('target_transpose_convolution', (2, 1, 1), 4, 2e-3, 2e-3),
        ('trilinear_upsample', (2, 1, 1), 4, 1e-5, 1e-6),
        ('sparse_reindex', (1, 1, 1), 3, 1e-6, 1e-6),
    ],
)
def test_torch_lattice_targeted_sparse_artifact_runs_on_mlx(
    case_name: str,
    source_stride: tuple[int, int, int],
    target_rows: int,
    rtol: float,
    atol: float,
) -> None:
    case = FIXTURE_ROOT / case_name
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))

    output = program(
        source=_sparse_input(
            inputs,
            prefix='source_',
            batch_counts=(int(inputs['source_active'].item()),),
            stride=source_stride,
        ),
        target=_sparse_input(
            inputs,
            prefix='target_',
            batch_counts=(target_rows,),
        ),
    )

    _assert_sparse_output_close(output, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ('case_name', 'batch_counts', 'input_prefix'),
    [
        ('transpose_convolution', (4,), 'x_'),
        ('generative_transpose_convolution', (2,), 'x_'),
        ('normalized_convolution', (4,), 'x_'),
    ],
)
def test_torch_lattice_transpose_artifact_runs_on_mlx(
    case_name: str,
    batch_counts: tuple[int, ...],
    input_prefix: str,
) -> None:
    case = FIXTURE_ROOT / case_name
    program = load_lattice_program(case)
    inputs = mx.load(str(case / 'inputs.safetensors'))
    expected = mx.load(str(case / 'expected.safetensors'))

    output = program(
        _sparse_input(
            inputs,
            batch_counts=batch_counts,
            prefix=input_prefix,
        )
    )

    _assert_sparse_output_close(output, expected, rtol=2e-3, atol=2e-3)


def _sparse_input(
    inputs: dict[str, mx.array],
    *,
    batch_counts: tuple[int, ...],
    prefix: str = '',
    stride: int | tuple[int, int, int] = 1,
) -> SparseTensor:
    return SparseTensor(
        inputs[f'{prefix}coords'],
        inputs[f'{prefix}features'],
        active_rows=inputs[f'{prefix}active'],
        batch_counts=batch_counts,
        stride=stride,
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

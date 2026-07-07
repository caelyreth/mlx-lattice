from __future__ import annotations

from pathlib import Path

import pytest
from lattice_contract import (
    LATTICE_DIALECT,
    MLIRModuleBuilder,
    SparseTensorType,
    TensorType,
    WeightType,
    dense_packing,
)

from mlx_lattice import SparseTensor
from mlx_lattice import _ext as ext
from mlx_lattice.artifact import (
    LatticeArtifact,
    compile_lattice_artifact,
    lattice_artifact_status,
    lattice_graph_operation_names,
    load_lattice_artifact,
    load_lattice_program,
    save_lattice_artifact,
    validate_lattice_artifact,
)
from mlx_lattice.artifact.lowering import ARTIFACT_LOWERINGS
from mlx_lattice.ops import conv3d
from tests.support import mx


def test_lattice_artifact_roundtrips_mlir_graph_and_weights(
    tmp_path,
) -> None:
    graph = _graph()
    weights = {'stem.weight': mx.ones((32, 3, 3, 3, 32), dtype=mx.float16)}

    save_lattice_artifact(tmp_path, graph, weights)
    artifact = load_lattice_artifact(tmp_path)

    assert isinstance(artifact, LatticeArtifact)
    assert artifact.graph == graph
    assert set(artifact.weights) == {'stem.weight'}
    assert artifact.weights['stem.weight'].shape == (32, 3, 3, 3, 32)


def test_lattice_artifact_requires_graph_mlir(tmp_path) -> None:
    mx.save_safetensors(str(tmp_path / 'weights.safetensors'), {})

    try:
        load_lattice_artifact(tmp_path)
    except ValueError as exc:
        assert 'graph.mlir' in str(exc)
    else:
        raise AssertionError('expected missing graph.mlir to fail')


def test_artifact_runtime_lowerings_cover_annotated_dialect_ops() -> None:
    assert set(ARTIFACT_LOWERINGS.functions) == {
        f'{LATTICE_DIALECT.namespace}.{op.name}'
        for op in LATTICE_DIALECT.iter_ops()
    }


def test_lattice_artifact_validates_with_mlir_tooling(tmp_path) -> None:
    if not _has_mlir_validator():
        pytest.skip('MLIR validator is not available in this environment.')
    graph = _graph()
    save_lattice_artifact(
        tmp_path,
        graph,
        {'stem.weight': mx.ones((32, 3, 3, 3, 32), dtype=mx.float16)},
    )

    validate_lattice_artifact(tmp_path)

    status = lattice_artifact_status(tmp_path)
    assert status.valid
    assert status.diagnostics == ''


def test_lattice_graph_operation_names_use_native_mlir_when_available() -> (
    None
):
    if not hasattr(ext, 'lattice_mlir_operation_names'):
        pytest.skip('MLIR-enabled native extension is not available.')

    assert lattice_graph_operation_names(_graph()) == (
        'lattice.sparse.make',
        'lattice.weight',
        'lattice.conv3d',
    )


def test_lattice_artifact_compiles_to_mlx_runtime_when_native_mlir_exists(
    tmp_path,
) -> None:
    if not hasattr(ext, 'lattice_mlir_plan'):
        pytest.skip('MLIR-enabled native extension is not available.')
    graph = _pointwise_graph()
    weight = mx.array(
        [[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]],
        dtype=mx.float16,
    )
    x = _input_tensor()
    save_lattice_artifact(tmp_path, graph, {'stem.weight': weight})

    actual = load_lattice_program(tmp_path)(x)
    expected = conv3d(x, weight, kernel_size=1)

    assert isinstance(actual, SparseTensor)
    assert actual.coords.tolist() == expected.coords.tolist()
    assert actual.active_rows.tolist() == expected.active_rows.tolist()
    assert mx.allclose(actual.feats, expected.feats)


def test_lattice_artifact_runtime_can_decompose_sparse_result() -> None:
    if not hasattr(ext, 'lattice_mlir_plan'):
        pytest.skip('MLIR-enabled native extension is not available.')
    x = _input_tensor()
    artifact = LatticeArtifact(_decompose_graph(), {})

    output = compile_lattice_artifact(artifact)(x)
    assert isinstance(output, tuple)
    coords, feats, active = output
    assert not isinstance(coords, SparseTensor)
    assert not isinstance(feats, SparseTensor)
    assert not isinstance(active, SparseTensor)

    assert coords.tolist() == x.coords.tolist()
    assert feats.tolist() == x.feats.tolist()
    assert active.tolist() == x.active_rows.tolist()


def _graph() -> str:
    sparse = SparseTensorType(dtype='f16')
    builder = MLIRModuleBuilder()
    coords = builder.argument('coords', TensorType('tensor<?x4xi32>'))
    feats = builder.argument('features', TensorType('tensor<?x32xf16>'))
    active = builder.argument('active', TensorType('tensor<1xi32>'))
    x = builder.sparse_make(
        coords=coords,
        features=feats,
        active=active,
        stride=(1, 1, 1),
        coord_order='batch_x_y_z',
        result_type=sparse,
    )
    weight = builder.weight(
        sym_name='stem.weight',
        storage_key='stem.weight',
        layout='conv3d_o_zyx_i',
        packing=dense_packing(),
        result_type=WeightType('conv3d', 'f16'),
    )
    out = builder.conv3d(
        input=x,
        weight=weight,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
        result_type=sparse,
    )
    builder.return_(out)
    return builder.to_mlir()


def _pointwise_graph() -> str:
    sparse = SparseTensorType(dtype='f16')
    builder = MLIRModuleBuilder()
    coords = builder.argument('coords', TensorType('tensor<?x4xi32>'))
    feats = builder.argument('features', TensorType('tensor<?x3xf16>'))
    active = builder.argument('active', TensorType('tensor<1xi32>'))
    x = builder.sparse_make(
        coords=coords,
        features=feats,
        active=active,
        stride=(1, 1, 1),
        coord_order='batch_x_y_z',
        result_type=sparse,
    )
    weight = builder.weight(
        sym_name='stem.weight',
        storage_key='stem.weight',
        layout='conv3d_o_zyx_i',
        packing=dense_packing(),
        result_type=WeightType('conv3d', 'f16'),
    )
    out = builder.conv3d(
        input=x,
        weight=weight,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        result_type=sparse,
    )
    builder.return_(out)
    return builder.to_mlir()


def _decompose_graph() -> str:
    sparse = SparseTensorType(dtype='f16')
    builder = MLIRModuleBuilder()
    coords = builder.argument('coords', TensorType('tensor<?x4xi32>'))
    feats = builder.argument('features', TensorType('tensor<?x3xf16>'))
    active = builder.argument('active', TensorType('tensor<1xi32>'))
    x = builder.sparse_make(
        coords=coords,
        features=feats,
        active=active,
        stride=(1, 1, 1),
        coord_order='batch_x_y_z',
        result_type=sparse,
    )
    out_coords, out_feats, out_active = builder.sparse_decompose(
        input=x,
        result_types=(
            TensorType('tensor<?x4xi32>'),
            TensorType('tensor<?x3xf16>'),
            TensorType('tensor<1xi32>'),
        ),
    )
    builder.return_(out_coords, out_feats, out_active)
    return builder.to_mlir()


def _input_tensor() -> SparseTensor:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=mx.float16,
    )
    return SparseTensor(coords, feats)


def _has_mlir_validator() -> bool:
    return (
        hasattr(ext, 'validate_lattice_mlir')
        or Path(
            'build/clangd-mlir/mlir/tools/lattice-opt/lattice-opt'
        ).is_file()
    )

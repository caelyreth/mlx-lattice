from __future__ import annotations

from lattice_contract import (
    MLIRModuleBuilder,
    SparseTensorType,
    TensorType,
    WeightType,
    dense_packing,
)

from mlx_lattice.artifact import (
    LatticeArtifact,
    load_lattice_artifact,
    save_lattice_artifact,
)
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

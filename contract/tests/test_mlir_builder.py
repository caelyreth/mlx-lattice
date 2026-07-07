from __future__ import annotations

from pathlib import Path
from subprocess import run

from lattice_contract import (
    LATTICE_DIALECT,
    MLIRModuleBuilder,
    SparseTensorType,
    TensorType,
    WeightType,
    dense_packing,
    quantized_packing,
)
from lattice_contract.mlir.builder import EMITTERS
from lattice_contract.schema import schema_digest


def test_lattice_dialect_schema_is_annotation_backed() -> None:
    digest = schema_digest(LATTICE_DIALECT)

    assert digest['types'] == ('SparseTensor', 'Weight')
    assert 'conv3d' in digest['ops']
    assert LATTICE_DIALECT.op_by_python_name('sparse_make').name == (
        'sparse.make'
    )
    assert LATTICE_DIALECT.attrs['Packing'].values == (
        'dense',
        'int4',
        'int8',
    )


def test_mlir_builder_emits_valid_conv3d_graph_shape() -> None:
    graph = _conv_graph()

    assert 'func.func @forward' in graph
    assert 'lattice.sparse.make' in graph
    assert 'lattice.weight @stem.weight' in graph
    assert 'lattice.conv3d' in graph
    assert 'array<i64: 3, 3, 3>' in graph
    assert '#lattice.packing<dense>' in graph


def test_mlir_builder_special_emitters_are_annotation_registered() -> None:
    assert set(EMITTERS.functions) == {
        LATTICE_DIALECT.op_by_python_name('sparse_decompose').name,
        LATTICE_DIALECT.op_by_python_name('weight').name,
    }


def test_mlir_builder_supports_quantized_weight_packing() -> None:
    builder = MLIRModuleBuilder()
    weight = builder.weight(
        sym_name='stem.qweight',
        storage_key='stem.qweight',
        layout='conv3d_o_zyx_i',
        packing=quantized_packing('int4', group_size=32),
        result_type=WeightType('conv3d', 'i4'),
    )
    builder.return_(weight)

    graph = builder.to_mlir()

    assert '#lattice.packing<int4' in graph
    assert 'group_size = 32' in graph
    assert '!lattice.weight<conv3d, i4>' in graph


def test_mlir_builder_output_passes_lattice_opt_when_available(
    tmp_path: Path,
) -> None:
    tool = Path('build/clangd-mlir/mlir/tools/lattice-opt/lattice-opt')
    if not tool.exists():
        return
    graph = tmp_path / 'graph.mlir'
    graph.write_text(_conv_graph(), encoding='utf-8')

    result = run(
        [str(tool), str(graph), '-o', str(tmp_path / 'out.mlir')],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _conv_graph() -> str:
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

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
    assert 'conv_transpose3d' in digest['ops']
    assert 'target_conv_transpose3d' in digest['ops']
    assert 'generative_conv_transpose3d' in digest['ops']
    assert 'normalized_subm_conv3d' in digest['ops']
    assert 'normalized_conv_transpose3d' in digest['ops']
    assert 'target_normalized_conv_transpose3d' in digest['ops']
    assert 'normalized_generative_conv_transpose3d' in digest['ops']
    assert 'pool3d' in digest['ops']
    assert 'pool_transpose3d' in digest['ops']
    assert 'trilinear_upsample3d' in digest['ops']
    assert 'global_pool' in digest['ops']
    assert 'voxelize' in digest['ops']
    assert 'devoxelize' in digest['ops']
    assert 'activation' in digest['ops']
    assert 'batch_norm' in digest['ops']
    assert 'layer_norm' in digest['ops']
    assert 'rms_norm' in digest['ops']
    assert 'sparse.binary' in digest['ops']
    assert 'sparse.cat' in digest['ops']
    assert 'sparse.reindex' in digest['ops']
    assert LATTICE_DIALECT.op_by_python_name('sparse_make').name == (
        'sparse.make'
    )
    assert LATTICE_DIALECT.qualified_op_name('sparse.make') == (
        'lattice.sparse.make'
    )
    assert LATTICE_DIALECT.resolve_op('lattice.sparse.make').name == (
        'sparse.make'
    )
    assert (
        LATTICE_DIALECT.resolve_qualified_op('lattice.conv3d').python_name
        == 'conv3d'
    )
    assert LATTICE_DIALECT.attrs['Packing'].values == (
        'dense',
        'int4',
        'int8',
    )
    assert tuple(
        param.name for param in LATTICE_DIALECT.attrs['Packing'].parameters
    ) == ('kind', 'group_size', 'scale_dtype', 'mode')
    assert LATTICE_DIALECT.attrs['Activation'].values == (
        'relu',
        'sigmoid',
        'gelu',
        'silu',
        'leaky_relu',
        'tanh',
        'softplus',
    )


def test_mlir_builder_emits_valid_conv3d_graph_shape() -> None:
    graph = _conv_graph()

    assert 'func.func @forward' in graph
    assert 'lattice.sparse.make' in graph
    assert 'lattice.weight @stem.weight' in graph
    assert 'lattice.conv3d' in graph
    assert 'array<i64: 3, 3, 3>' in graph
    assert '#lattice.packing<dense>' in graph


def test_mlir_builder_allocates_unique_explicit_ssa_names() -> None:
    sparse = SparseTensorType(dtype='f32')
    builder = MLIRModuleBuilder()
    coords = builder.argument('coords', TensorType('tensor<?x4xi32>'))
    features = builder.argument('features', TensorType('tensor<?x1xf32>'))
    active = builder.argument('active', TensorType('tensor<1xi32>'))
    first = builder.sparse_make(
        coords=coords,
        features=features,
        active=active,
        stride=(1, 1, 1),
        coord_order='batch_x_y_z',
        result_type=sparse,
        result='shared',
    )
    second = builder.sparse_with_features(
        input=first,
        features=features,
        result_type=sparse,
        result='shared',
    )
    builder.return_(second)

    graph = builder.to_mlir()

    assert '%shared = lattice.sparse.make' in graph
    assert '%shared1 = lattice.sparse.with_features' in graph


def test_mlir_builder_emits_transpose_conv_family_ops() -> None:
    graph = _transpose_conv_graph()

    assert 'lattice.conv_transpose3d' in graph
    assert 'lattice.generative_conv_transpose3d' in graph
    assert 'array<i64: 2, 2, 2>' in graph


def test_mlir_builder_emits_normalized_conv_family_ops() -> None:
    graph = _normalized_conv_graph()

    assert 'lattice.normalized_subm_conv3d' in graph
    assert 'lattice.normalized_conv_transpose3d' in graph
    assert 'lattice.normalized_generative_conv_transpose3d' in graph
    assert 'eps = 0.00000001 : f32' in graph


def test_mlir_builder_emits_target_transpose_conv_ops() -> None:
    sparse = SparseTensorType(dtype='f32')
    builder = MLIRModuleBuilder()
    source = builder.argument('source', sparse)
    target = builder.argument('target', sparse)
    weight = builder.weight(
        sym_name='up.weight',
        storage_key='up.weight',
        layout='conv3d_o_zyx_i',
        result_type=WeightType('conv3d', 'f32'),
    )
    ordinary = builder.target_conv_transpose3d(
        input=source,
        target=target,
        weight=weight,
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
        padding=(1, 0, 0),
        dilation=(1, 1, 1),
        result_type=sparse,
    )
    normalized = builder.target_normalized_conv_transpose3d(
        input=ordinary,
        target=target,
        weight=weight,
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
        padding=(1, 0, 0),
        dilation=(1, 1, 1),
        eps=1e-8,
        result_type=sparse,
    )
    builder.return_(normalized)

    graph = builder.to_mlir()

    assert 'lattice.target_conv_transpose3d' in graph
    assert 'lattice.target_normalized_conv_transpose3d' in graph


def test_mlir_builder_emits_pooling_ops() -> None:
    graph = _pool_graph()

    assert 'lattice.pool3d' in graph
    assert 'mode = #lattice.pool_mode<avg>' in graph
    assert 'lattice.global_pool' in graph
    assert 'batch_size = -1' in graph


def test_mlir_builder_emits_trilinear_upsample() -> None:
    sparse = SparseTensorType(dtype='f32')
    builder = MLIRModuleBuilder()
    source = builder.argument('source', sparse)
    target = builder.argument('target', sparse)
    out = builder.trilinear_upsample3d(
        input=source,
        target=target,
        stride=(2, 2, 2),
        result_type=sparse,
    )
    builder.return_(out)

    assert 'lattice.trilinear_upsample3d' in builder.to_mlir()


def test_mlir_builder_emits_point_voxel_ops() -> None:
    graph = _point_voxel_graph()

    assert 'lattice.voxelize' in graph
    assert 'reduction = #lattice.voxel_reduction<mean>' in graph
    assert 'array<f64: 0.1, 0.1, 0.1>' in graph
    assert 'lattice.devoxelize' in graph
    assert 'interpolation = #lattice.point_interpolation<linear>' in graph


def test_mlir_builder_emits_sparse_binary_ops() -> None:
    graph = _sparse_binary_graph()

    assert 'lattice.sparse.binary' in graph
    assert 'op = #lattice.binary_op<maximum>' in graph
    assert 'join = #lattice.join<inner>' in graph
    assert 'lattice.sparse.cat' in graph


def test_mlir_builder_emits_sparse_reindex() -> None:
    sparse = SparseTensorType(dtype='f32')
    builder = MLIRModuleBuilder()
    input = builder.argument('input', sparse)
    target = builder.argument('target', sparse)

    out = builder.sparse_reindex(
        input=input,
        target=target,
        fill=-1.5,
        result_type=sparse,
    )
    builder.return_(out)

    graph = builder.to_mlir()
    assert 'lattice.sparse.reindex %input, %target' in graph
    assert 'fill = -1.5 : f32' in graph


def test_mlir_builder_emits_dense_feature_ops_with_sparse_identity() -> (
    None
):
    graph = _feature_graph()

    assert 'lattice.sparse.decompose' in graph
    assert 'lattice.linear' in graph
    assert 'lattice.activation' in graph
    assert 'kind = #lattice.activation<gelu>' in graph
    assert 'approximate = #lattice.gelu_approx<tanh>' in graph
    assert 'lattice.batch_norm' in graph
    assert 'lattice.layer_norm' in graph
    assert 'lattice.rms_norm' in graph
    assert 'lattice.sparse.with_features' in graph


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


def test_mlir_builder_emits_optional_bias_operands() -> None:
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
        result_type=WeightType('conv3d', 'f16'),
    )
    bias = builder.weight(
        sym_name='stem.bias',
        storage_key='stem.bias',
        layout='bias_c',
        result_type=WeightType('bias', 'f16'),
    )
    out = builder.conv3d(
        input=x,
        weight=weight,
        bias=bias,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        result_type=sparse,
    )
    builder.return_(out)

    graph = builder.to_mlir()

    assert 'lattice.weight @stem.bias' in graph
    assert '!lattice.weight<bias, f16>' in graph
    assert 'lattice.conv3d %sparse_make, %weight, %weight1' in graph


def test_mlir_builder_output_passes_lattice_opt_when_available(
    tmp_path: Path,
) -> None:
    tool = Path('build/clangd-mlir/mlir/tools/lattice-opt/lattice-opt')
    if not tool.exists():
        return
    graph = tmp_path / 'graph.mlir'
    for index, text in enumerate(
        (
            _conv_graph(),
            _normalized_conv_graph(),
            _point_voxel_graph(),
            _sparse_binary_graph(),
            _feature_graph(),
        )
    ):
        graph.write_text(text, encoding='utf-8')
        result = run(
            [
                str(tool),
                '--lattice-verify-artifact',
                str(graph),
                '-o',
                str(tmp_path / f'out{index}.mlir'),
            ],
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


def _transpose_conv_graph() -> str:
    sparse = SparseTensorType(dtype='f16')
    builder = MLIRModuleBuilder()
    coords = builder.argument('coords', TensorType('tensor<?x4xi32>'))
    feats = builder.argument('features', TensorType('tensor<?x32xf16>'))
    active = builder.argument('active', TensorType('tensor<1xi32>'))
    x = builder.sparse_make(
        coords=coords,
        features=feats,
        active=active,
        stride=(2, 2, 2),
        coord_order='batch_x_y_z',
        result_type=sparse,
    )
    weight = builder.weight(
        sym_name='decoder.weight',
        storage_key='decoder.weight',
        layout='conv3d_o_zyx_i',
        packing=dense_packing(),
        result_type=WeightType('conv3d', 'f16'),
    )
    up = builder.conv_transpose3d(
        input=x,
        weight=weight,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        result_type=sparse,
    )
    out = builder.generative_conv_transpose3d(
        input=up,
        weight=weight,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        result_type=sparse,
    )
    builder.return_(out)
    return builder.to_mlir()


def _normalized_conv_graph() -> str:
    sparse = SparseTensorType(dtype='f16')
    builder = MLIRModuleBuilder()
    coords = builder.argument('coords', TensorType('tensor<?x4xi32>'))
    feats = builder.argument('features', TensorType('tensor<?x32xf16>'))
    active = builder.argument('active', TensorType('tensor<1xi32>'))
    x = builder.sparse_make(
        coords=coords,
        features=feats,
        active=active,
        stride=(4, 4, 4),
        coord_order='batch_x_y_z',
        result_type=sparse,
    )
    weight = builder.weight(
        sym_name='normalized.weight',
        storage_key='normalized.weight',
        layout='conv3d_o_zyx_i',
        result_type=WeightType('conv3d', 'f16'),
    )
    subm = builder.normalized_subm_conv3d(
        input=x,
        weight=weight,
        kernel_size=(3, 3, 3),
        dilation=(1, 1, 1),
        eps=1e-8,
        result_type=sparse,
    )
    transposed = builder.normalized_conv_transpose3d(
        input=subm,
        weight=weight,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        eps=1e-8,
        result_type=sparse,
    )
    out = builder.normalized_generative_conv_transpose3d(
        input=transposed,
        weight=weight,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        eps=1e-8,
        result_type=sparse,
    )
    builder.return_(out)
    return builder.to_mlir()


def _pool_graph() -> str:
    sparse = SparseTensorType(dtype='f32')
    builder = MLIRModuleBuilder()
    coords = builder.argument('coords', TensorType('tensor<?x4xi32>'))
    feats = builder.argument('features', TensorType('tensor<?x32xf32>'))
    active = builder.argument('active', TensorType('tensor<1xi32>'))
    x = builder.sparse_make(
        coords=coords,
        features=feats,
        active=active,
        stride=(1, 1, 1),
        coord_order='batch_x_y_z',
        result_type=sparse,
    )
    pooled = builder.pool3d(
        input=x,
        mode='avg',
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        result_type=sparse,
    )
    out = builder.global_pool(
        input=pooled,
        mode='sum',
        batch_size=-1,
        result_type=TensorType('tensor<?x32xf32>'),
    )
    builder.return_(out)
    return builder.to_mlir()


def _point_voxel_graph() -> str:
    sparse = SparseTensorType(dtype='f32')
    builder = MLIRModuleBuilder()
    points = builder.argument('points', TensorType('tensor<?x3xf32>'))
    feats = builder.argument('features', TensorType('tensor<?x16xf32>'))
    batches = builder.argument('batch_indices', TensorType('tensor<?xi32>'))
    active = builder.argument('active_rows', TensorType('tensor<1xi32>'))
    voxels = builder.voxelize(
        points=points,
        features=feats,
        batch_indices=batches,
        active_rows=active,
        voxel_size=(0.1, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
        reduction='mean',
        stride=(1, 1, 1),
        result_type=sparse,
    )
    out = builder.devoxelize(
        points=points,
        voxels=voxels,
        batch_indices=batches,
        point_active_rows=active,
        voxel_size=(0.1, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
        interpolation='linear',
        result_type=TensorType('tensor<?x16xf32>'),
    )
    builder.return_(out)
    return builder.to_mlir()


def _sparse_binary_graph() -> str:
    sparse = SparseTensorType(dtype='f16')
    builder = MLIRModuleBuilder()
    lhs_coords = builder.argument(
        'lhs_coords', TensorType('tensor<?x4xi32>')
    )
    lhs_feats = builder.argument(
        'lhs_features', TensorType('tensor<?x32xf16>')
    )
    lhs_active = builder.argument('lhs_active', TensorType('tensor<1xi32>'))
    rhs_coords = builder.argument(
        'rhs_coords', TensorType('tensor<?x4xi32>')
    )
    rhs_feats = builder.argument(
        'rhs_features', TensorType('tensor<?x32xf16>')
    )
    rhs_active = builder.argument('rhs_active', TensorType('tensor<1xi32>'))
    lhs = builder.sparse_make(
        coords=lhs_coords,
        features=lhs_feats,
        active=lhs_active,
        stride=(1, 1, 1),
        coord_order='batch_x_y_z',
        result_type=sparse,
    )
    rhs = builder.sparse_make(
        coords=rhs_coords,
        features=rhs_feats,
        active=rhs_active,
        stride=(1, 1, 1),
        coord_order='batch_x_y_z',
        result_type=sparse,
    )
    out = builder.sparse_binary(
        lhs=lhs,
        rhs=rhs,
        op='maximum',
        join='inner',
        lhs_fill=0.0,
        rhs_fill=0.0,
        result_type=sparse,
    )
    cat = builder.sparse_cat(
        lhs=out,
        rhs=rhs,
        join='inner',
        result_type=sparse,
    )
    builder.return_(cat)
    return builder.to_mlir()


def _feature_graph() -> str:
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
    _, feature_rows, _ = builder.sparse_decompose(
        input=x,
        result_types=(
            TensorType('tensor<?x4xi32>'),
            TensorType('tensor<?x32xf16>'),
            TensorType('tensor<1xi32>'),
        ),
    )
    weight = builder.weight(
        sym_name='head.weight',
        storage_key='head.weight',
        layout='linear_o_i',
        packing=dense_packing(),
        result_type=WeightType('linear', 'f16'),
    )
    scale = builder.weight(
        sym_name='head.scale',
        storage_key='head.scale',
        layout='channel_c',
        packing=dense_packing(),
        result_type=WeightType('channel', 'f16'),
    )
    bias = builder.weight(
        sym_name='head.bias',
        storage_key='head.bias',
        layout='bias_c',
        packing=dense_packing(),
        result_type=WeightType('bias', 'f16'),
    )
    mean = builder.weight(
        sym_name='head.mean',
        storage_key='head.mean',
        layout='channel_c',
        packing=dense_packing(),
        result_type=WeightType('channel', 'f16'),
    )
    var = builder.weight(
        sym_name='head.var',
        storage_key='head.var',
        layout='channel_c',
        packing=dense_packing(),
        result_type=WeightType('channel', 'f16'),
    )
    projected = builder.linear(
        input=feature_rows,
        weight=weight,
        result_type=TensorType('tensor<?x16xf16>'),
    )
    normalized = builder.batch_norm(
        input=projected,
        scale=scale,
        bias=bias,
        mean=mean,
        var=var,
        eps=1e-5,
        result_type=TensorType('tensor<?x16xf16>'),
    )
    layered = builder.layer_norm(
        input=normalized,
        scale=scale,
        bias=bias,
        eps=1e-5,
        result_type=TensorType('tensor<?x16xf16>'),
    )
    rms = builder.rms_norm(
        input=layered,
        scale=scale,
        eps=1e-5,
        result_type=TensorType('tensor<?x16xf16>'),
    )
    activated = builder.activation(
        input=rms,
        kind='gelu',
        approximate='tanh',
        alpha=0.01,
        beta=1.0,
        threshold=20.0,
        result_type=TensorType('tensor<?x16xf16>'),
    )
    out = builder.sparse_with_features(
        input=x,
        features=activated,
        result_type=SparseTensorType(dtype='f16'),
    )
    builder.return_(out)
    return builder.to_mlir()

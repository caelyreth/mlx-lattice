from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from lattice_contract import (
    ARTIFACT_GRAPH_FILE,
    ARTIFACT_WEIGHT_FILE,
    CURRENT_DIALECT_VERSION,
    DIALECT_SCHEMA_DIGEST,
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
    LatticeProgram,
    compile_lattice_artifact,
    lattice_artifact_status,
    lattice_graph_operation_names,
    load_lattice_artifact,
    load_lattice_program,
    native_artifact_execution_available,
    save_lattice_artifact,
    validate_lattice_artifact,
)
from mlx_lattice.artifact.lowering import (
    ARTIFACT_LOWERINGS,
    _compile_artifact_lowering,
)
from mlx_lattice.artifact.plan import RuntimePlan
from mlx_lattice.core import dequantize_weight, quantize_weight
from mlx_lattice.ops import (
    batch_norm,
    cat,
    conv3d,
    layer_norm,
    linear,
    relu,
    rms_norm,
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
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ARTIFACT_GRAPH_FILE,
        ARTIFACT_WEIGHT_FILE,
    ]


def test_lattice_artifact_requires_graph_mlir(tmp_path) -> None:
    mx.save_safetensors(str(tmp_path / 'weights.safetensors'), {})
    (tmp_path / 'manifest.json').write_text(
        '{"nodes": [{"op": "legacy.conv"}]}',
        encoding='utf-8',
    )

    try:
        load_lattice_artifact(tmp_path)
    except ValueError as exc:
        assert 'graph.mlir' in str(exc)
    else:
        raise AssertionError('expected missing graph.mlir to fail')


def test_lattice_artifact_requires_directory(tmp_path) -> None:
    path = tmp_path / 'artifact.lattice'
    path.write_text('', encoding='utf-8')

    with pytest.raises(ValueError, match='directory does not exist'):
        load_lattice_artifact(path)


def test_lattice_artifact_requires_weights_file(tmp_path) -> None:
    (tmp_path / ARTIFACT_GRAPH_FILE).write_text(_graph(), encoding='utf-8')

    with pytest.raises(ValueError, match=ARTIFACT_WEIGHT_FILE):
        load_lattice_artifact(tmp_path)


def test_artifact_runtime_lowerings_cover_dialect_schema_ops() -> None:
    assert set(ARTIFACT_LOWERINGS.functions) == {
        LATTICE_DIALECT.qualified_op_name(op)
        for op in LATTICE_DIALECT.iter_ops()
    }


def test_artifact_lowering_bindings_validate_against_schema() -> None:
    def invalid_attr(
        input: mx.array,
        *,
        missing: str,
    ) -> mx.array:
        del input
        del missing
        return mx.array([])

    with pytest.raises(ValueError, match='is not declared'):
        _compile_artifact_lowering(
            invalid_attr,
            LATTICE_DIALECT.resolve_op('activation'),
        )


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


def test_native_artifact_execution_capability_matches_extension() -> None:
    assert native_artifact_execution_available() is callable(
        getattr(ext, 'lattice_mlir_plan', None)
    )


def test_lattice_artifact_compile_requires_native_mlir_execution() -> None:
    if native_artifact_execution_available():
        pytest.skip('native MLIR artifact execution is available.')

    with pytest.raises(RuntimeError, match='MLIR-enabled'):
        compile_lattice_artifact(LatticeArtifact(_graph(), {}))


@pytest.mark.parametrize(
    ('graph', 'diagnostic'),
    [
        (
            lambda: _graph().replace('  lattice.ir_version = 0,\n', ''),
            'lattice.ir_version',
        ),
        (
            lambda: _graph().replace(
                'lattice.ir_version = 0',
                'lattice.ir_version = 1',
            ),
            'unsupported lattice.ir_version',
        ),
        (
            lambda: _graph().replace(
                f'  lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}",\n',
                '',
            ),
            'lattice.schema_digest',
        ),
        (
            lambda: _graph().replace(
                DIALECT_SCHEMA_DIGEST,
                '0' * len(DIALECT_SCHEMA_DIGEST),
            ),
            'unsupported lattice.schema_digest',
        ),
        (
            lambda: _graph().replace(
                '  lattice.input_names = ["coords", "features", "active"],\n',
                '',
            ),
            'lattice.input_names',
        ),
        (
            lambda: _graph().replace(
                '  lattice.output_roles = ["sparse_tensor"],\n',
                '  lattice.output_roles = ["backend_tensor_ops"],\n',
            ),
            'unsupported role',
        ),
        (
            lambda: _graph().replace(
                ',\n  lattice.weight_file = "weights.safetensors"\n',
                '',
            ),
            'lattice.weight_file',
        ),
        (
            lambda: _graph().replace(
                'lattice.weight_file = "weights.safetensors"',
                'lattice.weight_file = "model.safetensors"',
            ),
            'unsupported lattice.weight_file',
        ),
        (
            lambda: _no_entry_graph(),
            'exactly one func.func entry',
        ),
        (
            lambda: _graph().replace(
                'func.func @forward', 'func.func @predict'
            ),
            'must be named @forward',
        ),
        (
            lambda: _multiple_entry_graph(),
            'supports exactly one func.func entry',
        ),
        (
            lambda: _empty_return_graph(),
            'must return at least one value',
        ),
        (
            lambda: _non_lattice_body_graph(),
            'may contain only lattice operations and func.return',
        ),
    ],
)
def test_lattice_artifact_validation_enforces_module_contract(
    graph,
    diagnostic: str,
) -> None:
    if not _has_mlir_validator():
        pytest.skip('MLIR validator is not available in this environment.')

    status = lattice_artifact_status(LatticeArtifact(graph(), {}))

    assert not status.valid
    assert diagnostic in status.diagnostics


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


def test_native_lattice_mlir_plan_exposes_typed_abi_metadata() -> None:
    if not hasattr(ext, 'lattice_mlir_plan'):
        pytest.skip('MLIR-enabled native extension is not available.')

    plan = ext.lattice_mlir_plan(_pointwise_graph())
    plan_ops = cast(list[dict[str, object]], plan['ops'])

    assert plan['ir_version'] == CURRENT_DIALECT_VERSION
    assert plan['schema_digest'] == DIALECT_SCHEMA_DIGEST
    assert plan['weight_file'] == ARTIFACT_WEIGHT_FILE
    assert plan['args'] == [
        {
            'name': 'arg0',
            'abi_name': 'coords',
            'type': 'tensor<?x4xi32>',
            'role': 'sparse_coords',
        },
        {
            'name': 'arg1',
            'abi_name': 'features',
            'type': 'tensor<?x3xf16>',
            'role': 'sparse_features',
        },
        {
            'name': 'arg2',
            'abi_name': 'active',
            'type': 'tensor<1xi32>',
            'role': 'sparse_active',
        },
    ]
    assert plan_ops[0]['name'] == 'lattice.sparse.make'
    assert plan_ops[0]['operand_types'] == [
        'tensor<?x4xi32>',
        'tensor<?x3xf16>',
        'tensor<1xi32>',
    ]
    assert plan_ops[0]['result_types'] == [
        '!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, '
        'feature = row_channel, dtype = f16>'
    ]
    assert plan['outputs'] == [
        {
            'name': 'v5',
            'abi_name': 'conv3d',
            'type': '!lattice.sparse_tensor<rank = 3, '
            'coord = batch_x_y_z, feature = row_channel, dtype = f16>',
            'role': 'sparse_tensor',
        }
    ]


def test_runtime_plan_freezes_native_payload() -> None:
    plan = _runtime_plan(_activation_plan(kind='relu'))

    assert plan.ir_version == CURRENT_DIALECT_VERSION
    assert plan.schema_digest == DIALECT_SCHEMA_DIGEST
    assert plan.weight_file == ARTIFACT_WEIGHT_FILE
    assert plan.name == 'forward'
    assert [argument.abi_name for argument in plan.args] == [
        'coords',
        'features',
        'active',
    ]
    assert [argument.role for argument in plan.args] == [
        'sparse_coords',
        'sparse_features',
        'sparse_active',
    ]
    assert plan.ops[0].name == 'lattice.sparse.make'
    assert plan.ops[0].definition.name == 'sparse.make'
    assert plan.ops[0].operands == ('arg0', 'arg1', 'arg2')
    assert plan.ops[0].attrs['coord_order'] == 'batch_x_y_z'
    assert plan.ops[0].attrs['stride'] == (1, 1, 1)
    with pytest.raises(TypeError):
        cast(dict[str, object], plan.ops[0].attrs)['coord_order'] = 'bad'
    with pytest.raises(TypeError):
        cast(list[int], plan.ops[0].attrs['stride'])[0] = 2


def test_runtime_plan_deep_freezes_nested_attrs() -> None:
    plan = _runtime_plan(_quantized_conv_plan(bits=4))
    packing = plan.ops[1].attrs['packing']

    assert packing['kind'] == 'int4'
    with pytest.raises(TypeError):
        cast(dict[str, object], packing)['kind'] = 'int8'


def test_runtime_plan_rejects_mismatched_type_metadata() -> None:
    raw = _activation_plan(kind='relu')
    ops = raw['ops']
    assert isinstance(ops, list)
    first_op = ops[0]
    assert isinstance(first_op, dict)
    cast(dict[str, object], first_op)['operand_types'] = ['tensor<?x4xi32>']

    with pytest.raises(ValueError, match='operand_types'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_missing_module_metadata() -> None:
    raw = _activation_plan(kind='relu')
    del raw['ir_version']

    with pytest.raises(KeyError, match='ir_version'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_unsupported_schema_digest() -> None:
    raw = _activation_plan(kind='relu')
    raw['schema_digest'] = '0' * len(DIALECT_SCHEMA_DIGEST)

    with pytest.raises(ValueError, match='schema_digest'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_unsupported_module_metadata() -> None:
    raw = _activation_plan(kind='relu')
    raw['weight_file'] = 'model.safetensors'

    with pytest.raises(ValueError, match='weight_file'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_unknown_operation_name() -> None:
    raw = _activation_plan(kind='relu')
    ops = raw['ops']
    assert isinstance(ops, list)
    first_op = ops[0]
    assert isinstance(first_op, dict)
    cast(dict[str, object], first_op)['name'] = 'lattice.nope'

    with pytest.raises(ValueError, match='unknown lattice operation'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_schema_arity_mismatch() -> None:
    raw = _activation_plan(kind='relu')
    ops = raw['ops']
    assert isinstance(ops, list)
    first_op = ops[0]
    assert isinstance(first_op, dict)
    cast(dict[str, object], first_op)['operands'] = ['arg0']

    with pytest.raises(ValueError, match='SSA operands'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_schema_attr_mismatch() -> None:
    raw = _activation_plan(kind='relu')
    ops = raw['ops']
    assert isinstance(ops, list)
    first_op = ops[0]
    assert isinstance(first_op, dict)
    cast(dict[str, object], first_op)['attrs'] = {'stride': [1, 1, 1]}

    with pytest.raises(ValueError, match='missing required attrs'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_backend_route_attrs() -> None:
    raw = _activation_plan(kind='relu')
    ops = raw['ops']
    assert isinstance(ops, list)
    activation_op = cast(dict[str, object], ops[2])
    attrs = cast(dict[str, object], activation_op['attrs'])
    attrs['tensor_ops_route'] = 'm5_row_stationary'

    with pytest.raises(ValueError, match='unknown attrs'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_invalid_enum_attr_value() -> None:
    raw = _activation_plan(kind='mish')

    with pytest.raises(ValueError, match='kind'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_invalid_triple_attr_shape() -> None:
    raw = _activation_plan(kind='relu')
    ops = raw['ops']
    assert isinstance(ops, list)
    first_op = cast(dict[str, object], ops[0])
    attrs = cast(dict[str, object], first_op['attrs'])
    attrs['stride'] = [1, 1]

    with pytest.raises(ValueError, match='stride'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_invalid_packing_attr() -> None:
    raw = _quantized_conv_plan(bits=4)
    ops = raw['ops']
    assert isinstance(ops, list)
    weight_op = cast(dict[str, object], ops[1])
    attrs = cast(dict[str, object], weight_op['attrs'])
    packing = cast(dict[str, object], attrs['packing'])
    packing['scale_dtype'] = 'bf16'

    with pytest.raises(ValueError, match='scale_dtype'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_wrong_entry_name() -> None:
    raw = _activation_plan(kind='relu')
    raw['name'] = 'predict'

    with pytest.raises(ValueError, match='entry must be forward'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_duplicate_value_labels() -> None:
    raw = _activation_plan(kind='relu')
    ops = raw['ops']
    assert isinstance(ops, list)
    second_op = ops[1]
    assert isinstance(second_op, dict)
    cast(dict[str, object], second_op)['results'] = ['v0', 'v2', 'v3']

    with pytest.raises(ValueError, match='duplicate runtime value label'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_undefined_operand() -> None:
    raw = _activation_plan(kind='relu')
    ops = raw['ops']
    assert isinstance(ops, list)
    second_op = ops[1]
    assert isinstance(second_op, dict)
    cast(dict[str, object], second_op)['operands'] = ['missing']

    with pytest.raises(ValueError, match='uses undefined value'):
        RuntimePlan.from_native(raw)


def test_runtime_plan_rejects_undefined_return() -> None:
    raw = _activation_plan(kind='relu')
    raw['returns'] = ['missing']

    with pytest.raises(ValueError, match='return uses undefined value'):
        RuntimePlan.from_native(raw)


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
    assert isinstance(coords, mx.array)
    assert isinstance(feats, mx.array)
    assert isinstance(active, mx.array)

    assert coords.tolist() == x.coords.tolist()
    assert feats.tolist() == x.feats.tolist()
    assert active.tolist() == x.active_rows.tolist()


def test_lattice_artifact_runtime_lowers_conv_bias() -> None:
    x = _input_tensor()
    weight = mx.array(
        [[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]],
        dtype=mx.float16,
    ).reshape((2, 1, 1, 1, x.channels))
    bias = mx.array([0.25, -0.5], dtype=mx.float16)

    actual = LatticeProgram(
        _runtime_plan(
            _dense_weight_plan(
                op_name='lattice.conv3d',
                weight_key='stem.weight',
                weight_layout='conv3d_o_zyx_i',
                bias_key='stem.bias',
                result_name='v3',
                attrs={
                    'kernel_size': [1, 1, 1],
                    'stride': [1, 1, 1],
                    'padding': [0, 0, 0],
                    'dilation': [1, 1, 1],
                },
            )
        ),
        {'stem.weight': weight, 'stem.bias': bias},
    )(x)
    expected = conv3d(x, weight, bias, kernel_size=1)

    assert isinstance(actual, SparseTensor)
    mx.eval(actual.feats, expected.feats)
    assert bool(mx.allclose(actual.feats, expected.feats))


def test_lattice_artifact_runtime_lowers_linear_bias() -> None:
    x = _input_tensor()
    weight = mx.array(
        [[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]],
        dtype=mx.float16,
    )
    bias = mx.array([0.25, -0.5], dtype=mx.float16)

    actual = LatticeProgram(
        _runtime_plan(
            _dense_linear_weight_plan(
                weight_key='head.weight',
                bias_key='head.bias',
            )
        ),
        {'head.weight': weight, 'head.bias': bias},
    )(x)
    expected = linear(x, weight, bias)

    assert isinstance(actual, SparseTensor)
    mx.eval(actual.feats, expected.feats)
    assert bool(mx.allclose(actual.feats, expected.feats))


def test_lattice_artifact_runtime_lowers_activation_with_sparse_identity() -> (
    None
):
    x = _input_tensor()

    actual = LatticeProgram(
        _runtime_plan(_activation_plan(kind='relu')), {}
    )(x)
    expected = relu(x)

    assert isinstance(actual, SparseTensor)
    assert actual.coords.tolist() == expected.coords.tolist()
    mx.eval(actual.feats, expected.feats)
    assert bool(mx.allclose(actual.feats, expected.feats))


def test_lattice_artifact_runtime_lowers_sparse_cat() -> None:
    x = _input_tensor()

    actual = LatticeProgram(_runtime_plan(_sparse_cat_plan()), {})(x)
    expected = cat((x, x))

    assert isinstance(actual, SparseTensor)
    assert actual.coords.tolist() == expected.coords.tolist()
    mx.eval(actual.feats, expected.feats)
    assert bool(mx.allclose(actual.feats, expected.feats))


def test_lattice_artifact_runtime_binds_stable_keyword_abi() -> None:
    x = _input_tensor()

    actual = LatticeProgram(
        _runtime_plan(_activation_plan(kind='relu')), {}
    )(
        coords=x.coords,
        features=x.feats,
        active=x.active_rows,
    )
    expected = relu(x)

    assert isinstance(actual, SparseTensor)
    mx.eval(actual.feats, expected.feats)
    assert bool(mx.allclose(actual.feats, expected.feats))


def test_lattice_artifact_runtime_binds_positional_abi() -> None:
    x = _input_tensor()

    actual = LatticeProgram(
        _runtime_plan(_activation_plan(kind='relu')), {}
    )(x.coords, x.feats, x.active_rows)
    expected = relu(x)

    assert isinstance(actual, SparseTensor)
    mx.eval(actual.feats, expected.feats)
    assert bool(mx.allclose(actual.feats, expected.feats))


def test_lattice_artifact_runtime_binds_multiple_sparse_inputs_and_outputs() -> (
    None
):
    x = _input_tensor()
    target = x.replace(feats=x.feats * 2)
    program = LatticeProgram(_runtime_plan(_two_sparse_input_plan()), {})

    positional = program(x, target)
    named = program(x=x, target=target)

    assert isinstance(positional, tuple)
    assert isinstance(named, tuple)
    assert len(positional) == len(named) == 2
    for actual, expected in zip(positional, (x, target), strict=True):
        assert isinstance(actual, SparseTensor)
        mx.eval(actual.feats, expected.feats)
        assert bool(mx.allclose(actual.feats, expected.feats))
    for actual, expected in zip(named, positional, strict=True):
        assert isinstance(actual, SparseTensor)
        assert isinstance(expected, SparseTensor)
        mx.eval(actual.feats, expected.feats)
        assert bool(mx.allclose(actual.feats, expected.feats))


def test_lattice_artifact_runtime_rejects_missing_keyword_input() -> None:
    x = _input_tensor()
    program = LatticeProgram(
        _runtime_plan(_activation_plan(kind='relu')), {}
    )

    with pytest.raises(ValueError, match='missing artifact input: active'):
        program(coords=x.coords, features=x.feats)


def test_lattice_artifact_runtime_rejects_unexpected_keyword_input() -> (
    None
):
    x = _input_tensor()
    program = LatticeProgram(
        _runtime_plan(_activation_plan(kind='relu')), {}
    )

    with pytest.raises(
        ValueError, match='unexpected artifact inputs: extra'
    ):
        program(
            coords=x.coords,
            features=x.feats,
            active=x.active_rows,
            extra=x.feats,
        )


def test_lattice_artifact_runtime_rejects_too_many_positional_inputs() -> (
    None
):
    x = _input_tensor()
    program = LatticeProgram(
        _runtime_plan(_activation_plan(kind='relu')), {}
    )

    with pytest.raises(ValueError, match='too many positional'):
        program(x.coords, x.feats, x.active_rows, x.feats)


def test_lattice_artifact_runtime_rejects_sparse_shorthand_with_kwargs() -> (
    None
):
    x = _input_tensor()
    program = LatticeProgram(
        _runtime_plan(_activation_plan(kind='relu')), {}
    )

    with pytest.raises(ValueError, match='cannot be combined'):
        program(x, active=x.active_rows)


def test_lattice_artifact_runtime_rejects_sparse_shorthand_role_drift() -> (
    None
):
    raw = _activation_plan(kind='relu')
    args = raw['args']
    assert isinstance(args, list)
    first_arg = cast(dict[str, object], args[0])
    first_arg['role'] = 'tensor'
    program = LatticeProgram(_runtime_plan(raw), {})

    with pytest.raises(ValueError, match='SparseTensor shorthand requires'):
        program(_input_tensor())


def test_lattice_artifact_runtime_rejects_invalid_input_value_type() -> (
    None
):
    x = _input_tensor()
    program = LatticeProgram(
        _runtime_plan(_activation_plan(kind='relu')), {}
    )

    with pytest.raises(TypeError, match='artifact inputs'):
        program(coords=x.coords, features=x.feats, active=object())


def test_lattice_artifact_runtime_lowers_norms_with_sparse_identity() -> (
    None
):
    x = _input_tensor()
    scale = mx.array([1.5, 0.5, 2.0], dtype=mx.float16)
    bias = mx.array([0.25, -0.5, 0.75], dtype=mx.float16)
    mean = mx.array([1.0, 2.0, 3.0], dtype=mx.float16)
    var = mx.array([0.5, 1.5, 2.0], dtype=mx.float16)
    weights = {
        'norm.scale': scale,
        'norm.bias': bias,
        'norm.mean': mean,
        'norm.var': var,
    }

    bn = LatticeProgram(_runtime_plan(_norm_plan('batch_norm')), weights)(x)
    ln = LatticeProgram(_runtime_plan(_norm_plan('layer_norm')), weights)(x)
    rms = LatticeProgram(_runtime_plan(_norm_plan('rms_norm')), weights)(x)

    assert isinstance(bn, SparseTensor)
    assert isinstance(ln, SparseTensor)
    assert isinstance(rms, SparseTensor)
    assert bn.coords.tolist() == x.coords.tolist()
    assert ln.coords.tolist() == x.coords.tolist()
    assert rms.coords.tolist() == x.coords.tolist()
    mx.eval(bn.feats, ln.feats, rms.feats)
    assert bool(
        mx.allclose(
            bn.feats,
            batch_norm(
                x,
                weight=scale,
                bias=bias,
                mean=mean,
                var=var,
            ).feats,
            rtol=1e-3,
            atol=1e-3,
        )
    )
    assert bool(
        mx.allclose(
            ln.feats,
            layer_norm(x, weight=scale, bias=bias).feats,
            rtol=1e-3,
            atol=1e-3,
        )
    )
    assert bool(
        mx.allclose(
            rms.feats,
            rms_norm(x, weight=scale).feats,
            rtol=1e-3,
            atol=1e-3,
        )
    )


@pytest.mark.parametrize('bits', [4, 8])
def test_lattice_artifact_runtime_resolves_quantized_conv_weights(
    bits: int,
) -> None:
    x = _input_tensor()
    dense = mx.array(
        [((index % 17) - 8) / 17 for index in range(4 * x.channels)],
        dtype=mx.float16,
    ).reshape((4, 1, 1, 1, x.channels))
    packed = quantize_weight(dense, bits=bits)

    actual = LatticeProgram(
        _runtime_plan(_quantized_conv_plan(bits=bits)),
        _packed_weights('stem.qweight', packed),
    )(x)
    expected = conv3d(x, dequantize_weight(packed), kernel_size=1)

    assert isinstance(actual, SparseTensor)
    mx.eval(actual.feats, expected.feats)
    assert actual.coords.tolist() == expected.coords.tolist()
    assert bool(
        mx.allclose(actual.feats, expected.feats, rtol=2e-2, atol=5e-3)
    )


@pytest.mark.parametrize('bits', [4, 8])
def test_lattice_artifact_runtime_resolves_quantized_linear_weights(
    bits: int,
) -> None:
    x = _input_tensor()
    dense = mx.array(
        [((index % 19) - 9) / 19 for index in range(5 * x.channels)],
        dtype=mx.float16,
    ).reshape((5, x.channels))
    packed = quantize_weight(dense, bits=bits)

    actual = LatticeProgram(
        _runtime_plan(_quantized_linear_plan(bits=bits)),
        _packed_weights('head.qweight', packed),
    )(x)
    expected = linear(x, dequantize_weight(packed))

    assert isinstance(actual, SparseTensor)
    mx.eval(actual.feats, expected.feats)
    assert actual.coords.tolist() == expected.coords.tolist()
    assert bool(
        mx.allclose(actual.feats, expected.feats, rtol=2e-2, atol=5e-3)
    )


def test_lattice_artifact_runtime_rejects_missing_dense_weight() -> None:
    with pytest.raises(ValueError, match='artifact weight not found'):
        LatticeProgram(
            _runtime_plan(
                _dense_linear_weight_plan(
                    weight_key='head.weight',
                    bias_key='head.bias',
                )
            ),
            {},
        )(_input_tensor())


def test_lattice_artifact_runtime_rejects_incomplete_quantized_weight() -> (
    None
):
    packed = quantize_weight(mx.ones((5, 3), dtype=mx.float16), bits=4)

    with pytest.raises(
        ValueError, match='quantized artifact weight requires'
    ):
        LatticeProgram(
            _runtime_plan(_quantized_linear_plan(bits=4)),
            {'head.qweight.weight': packed.weight},
        )(_input_tensor())


def test_lattice_artifact_runtime_rejects_quantized_scale_dtype_drift() -> (
    None
):
    packed = quantize_weight(mx.ones((5, 3), dtype=mx.float16), bits=4)
    weights = _packed_weights('head.qweight', packed)
    weights['head.qweight.scales'] = weights['head.qweight.scales'].astype(
        mx.float32
    )

    with pytest.raises(ValueError, match='scales dtype mismatch'):
        LatticeProgram(
            _runtime_plan(_quantized_linear_plan(bits=4)),
            weights,
        )(_input_tensor())


def _graph() -> str:
    sparse = SparseTensorType(dtype='f16')
    builder = MLIRModuleBuilder()
    coords = builder.argument(
        'coords',
        TensorType('tensor<?x4xi32>'),
        role='sparse_coords',
    )
    feats = builder.argument(
        'features',
        TensorType('tensor<?x32xf16>'),
        role='sparse_features',
    )
    active = builder.argument(
        'active',
        TensorType('tensor<1xi32>'),
        role='sparse_active',
    )
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


def _no_entry_graph() -> str:
    return (
        'module attributes {\n'
        '  lattice.ir_version = 0,\n'
        f'  lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}",\n'
        '  lattice.input_names = [],\n'
        '  lattice.input_roles = [],\n'
        '  lattice.output_names = ["output"],\n'
        '  lattice.output_roles = ["tensor"],\n'
        '  lattice.weight_file = "weights.safetensors"\n'
        '} {\n'
        '}\n'
    )


def _multiple_entry_graph() -> str:
    return (
        'module attributes {\n'
        '  lattice.ir_version = 0,\n'
        f'  lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}",\n'
        '  lattice.input_names = ["x"],\n'
        '  lattice.input_roles = ["tensor"],\n'
        '  lattice.output_names = ["output"],\n'
        '  lattice.output_roles = ["tensor"],\n'
        '  lattice.weight_file = "weights.safetensors"\n'
        '} {\n'
        '  func.func @forward(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {\n'
        '    return %x : tensor<?x4xi32>\n'
        '  }\n'
        '  func.func @aux(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {\n'
        '    return %x : tensor<?x4xi32>\n'
        '  }\n'
        '}\n'
    )


def _empty_return_graph() -> str:
    return (
        'module attributes {\n'
        '  lattice.ir_version = 0,\n'
        f'  lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}",\n'
        '  lattice.input_names = [],\n'
        '  lattice.input_roles = [],\n'
        '  lattice.output_names = [],\n'
        '  lattice.output_roles = [],\n'
        '  lattice.weight_file = "weights.safetensors"\n'
        '} {\n'
        '  func.func @forward() {\n'
        '    return\n'
        '  }\n'
        '}\n'
    )


def _non_lattice_body_graph() -> str:
    return (
        'module attributes {\n'
        '  lattice.ir_version = 0,\n'
        f'  lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}",\n'
        '  lattice.input_names = ["x"],\n'
        '  lattice.input_roles = ["tensor"],\n'
        '  lattice.output_names = ["output"],\n'
        '  lattice.output_roles = ["tensor"],\n'
        '  lattice.weight_file = "weights.safetensors"\n'
        '} {\n'
        '  func.func @forward(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {\n'
        '    %out = func.call @forward(%x) : '
        '(tensor<?x4xi32>) -> tensor<?x4xi32>\n'
        '    return %out : tensor<?x4xi32>\n'
        '  }\n'
        '}\n'
    )


def _pointwise_graph() -> str:
    sparse = SparseTensorType(dtype='f16')
    builder = MLIRModuleBuilder()
    coords = builder.argument(
        'coords',
        TensorType('tensor<?x4xi32>'),
        role='sparse_coords',
    )
    feats = builder.argument(
        'features',
        TensorType('tensor<?x3xf16>'),
        role='sparse_features',
    )
    active = builder.argument(
        'active',
        TensorType('tensor<1xi32>'),
        role='sparse_active',
    )
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
    coords = builder.argument(
        'coords',
        TensorType('tensor<?x4xi32>'),
        role='sparse_coords',
    )
    feats = builder.argument(
        'features',
        TensorType('tensor<?x3xf16>'),
        role='sparse_features',
    )
    active = builder.argument(
        'active',
        TensorType('tensor<1xi32>'),
        role='sparse_active',
    )
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


def _runtime_plan(raw: dict[str, object]) -> RuntimePlan:
    return RuntimePlan.from_native(raw)


def _plan_payload(
    *,
    ops: list[dict[str, object]],
    returns: list[str],
) -> dict[str, object]:
    return {
        'ir_version': CURRENT_DIALECT_VERSION,
        'schema_digest': DIALECT_SCHEMA_DIGEST,
        'weight_file': ARTIFACT_WEIGHT_FILE,
        'name': 'forward',
        'args': _sparse_tensor_plan_args(),
        'ops': ops,
        'returns': returns,
        'outputs': [
            {
                'name': value,
                'abi_name': 'output' if index == 0 else f'output{index}',
                'type': 'unknown',
                'role': 'sparse_tensor',
            }
            for index, value in enumerate(returns)
        ],
    }


def _sparse_tensor_plan_args() -> list[dict[str, str]]:
    return [
        {
            'name': 'arg0',
            'abi_name': 'coords',
            'type': 'tensor<?x4xi32>',
            'role': 'sparse_coords',
        },
        {
            'name': 'arg1',
            'abi_name': 'features',
            'type': 'tensor<?x3xf16>',
            'role': 'sparse_features',
        },
        {
            'name': 'arg2',
            'abi_name': 'active',
            'type': 'tensor<1xi32>',
            'role': 'sparse_active',
        },
    ]


def _two_sparse_input_plan() -> dict[str, object]:
    args: list[dict[str, str]] = []
    for offset, prefix in ((0, 'x'), (3, 'target')):
        args.extend(
            [
                {
                    'name': f'arg{offset}',
                    'abi_name': f'{prefix}_coords',
                    'type': 'tensor<?x4xi32>',
                    'role': 'sparse_coords',
                },
                {
                    'name': f'arg{offset + 1}',
                    'abi_name': f'{prefix}_features',
                    'type': 'tensor<?x3xf16>',
                    'role': 'sparse_features',
                },
                {
                    'name': f'arg{offset + 2}',
                    'abi_name': f'{prefix}_active',
                    'type': 'tensor<1xi32>',
                    'role': 'sparse_active',
                },
            ]
        )
    return {
        'ir_version': CURRENT_DIALECT_VERSION,
        'schema_digest': DIALECT_SCHEMA_DIGEST,
        'weight_file': ARTIFACT_WEIGHT_FILE,
        'name': 'forward',
        'args': args,
        'ops': [
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg0', 'arg1', 'arg2'],
                'results': ['x'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg3', 'arg4', 'arg5'],
                'results': ['target'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
        ],
        'returns': ['x', 'target'],
        'outputs': [
            {
                'name': 'x',
                'abi_name': 'features',
                'type': '!lattice.sparse_tensor<f16>',
                'role': 'sparse_tensor',
            },
            {
                'name': 'target',
                'abi_name': 'target_features',
                'type': '!lattice.sparse_tensor<f16>',
                'role': 'sparse_tensor',
            },
        ],
    }


def _quantized_conv_plan(*, bits: int) -> dict[str, object]:
    return _single_sparse_input_plan(
        weight_key='stem.qweight',
        weight_layout='conv3d_o_zyx_i',
        weight_family_op='lattice.conv3d',
        packing_kind='int4' if bits == 4 else 'int8',
        result_name='v2',
        consumer_attrs={
            'kernel_size': [1, 1, 1],
            'stride': [1, 1, 1],
            'padding': [0, 0, 0],
            'dilation': [1, 1, 1],
        },
    )


def _quantized_linear_plan(*, bits: int) -> dict[str, object]:
    return _linear_sparse_identity_plan(
        weight_key='head.qweight',
        packing_kind='int4' if bits == 4 else 'int8',
    )


def _single_sparse_input_plan(
    *,
    weight_key: str,
    weight_layout: str,
    weight_family_op: str,
    packing_kind: str,
    result_name: str,
    consumer_attrs: dict[str, object],
) -> dict[str, object]:
    return _plan_payload(
        ops=[
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg0', 'arg1', 'arg2'],
                'results': ['v0'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
            {
                'name': 'lattice.weight',
                'operands': [],
                'results': ['v1'],
                'attrs': {
                    'storage_key': weight_key,
                    'layout': weight_layout,
                    'packing': {
                        'kind': packing_kind,
                        'group_size': 32,
                        'scale_dtype': 'f16',
                        'mode': 'affine',
                    },
                },
            },
            {
                'name': weight_family_op,
                'operands': ['v0', 'v1'],
                'results': [result_name],
                'attrs': consumer_attrs,
            },
        ],
        returns=[result_name],
    )


def _dense_weight_plan(
    *,
    op_name: str,
    weight_key: str,
    weight_layout: str,
    bias_key: str,
    result_name: str,
    attrs: dict[str, object],
) -> dict[str, object]:
    return _plan_payload(
        ops=[
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg0', 'arg1', 'arg2'],
                'results': ['v0'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
            {
                'name': 'lattice.weight',
                'operands': [],
                'results': ['v1'],
                'attrs': {
                    'storage_key': weight_key,
                    'layout': weight_layout,
                    'packing': {'kind': 'dense'},
                },
            },
            {
                'name': 'lattice.weight',
                'operands': [],
                'results': ['v2'],
                'attrs': {
                    'storage_key': bias_key,
                    'layout': 'bias_c',
                    'packing': {'kind': 'dense'},
                },
            },
            {
                'name': op_name,
                'operands': ['v0', 'v1', 'v2'],
                'results': [result_name],
                'attrs': attrs,
            },
        ],
        returns=[result_name],
    )


def _dense_linear_weight_plan(
    *,
    weight_key: str,
    bias_key: str,
) -> dict[str, object]:
    return _plan_payload(
        ops=[
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg0', 'arg1', 'arg2'],
                'results': ['v0'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
            {
                'name': 'lattice.sparse.decompose',
                'operands': ['v0'],
                'results': ['v1', 'v2', 'v3'],
                'attrs': {},
            },
            {
                'name': 'lattice.weight',
                'operands': [],
                'results': ['v4'],
                'attrs': {
                    'storage_key': weight_key,
                    'layout': 'linear_o_i',
                    'packing': {'kind': 'dense'},
                },
            },
            {
                'name': 'lattice.weight',
                'operands': [],
                'results': ['v5'],
                'attrs': {
                    'storage_key': bias_key,
                    'layout': 'bias_c',
                    'packing': {'kind': 'dense'},
                },
            },
            {
                'name': 'lattice.linear',
                'operands': ['v2', 'v4', 'v5'],
                'results': ['v6'],
                'attrs': {},
            },
            {
                'name': 'lattice.sparse.with_features',
                'operands': ['v0', 'v6'],
                'results': ['v7'],
                'attrs': {},
            },
        ],
        returns=['v7'],
    )


def _linear_sparse_identity_plan(
    *,
    weight_key: str,
    packing_kind: str,
) -> dict[str, object]:
    return _plan_payload(
        ops=[
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg0', 'arg1', 'arg2'],
                'results': ['v0'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
            {
                'name': 'lattice.sparse.decompose',
                'operands': ['v0'],
                'results': ['v1', 'v2', 'v3'],
                'attrs': {},
            },
            {
                'name': 'lattice.weight',
                'operands': [],
                'results': ['v4'],
                'attrs': {
                    'storage_key': weight_key,
                    'layout': 'linear_o_i',
                    'packing': {
                        'kind': packing_kind,
                        'group_size': 32,
                        'scale_dtype': 'f16',
                        'mode': 'affine',
                    },
                },
            },
            {
                'name': 'lattice.linear',
                'operands': ['v2', 'v4'],
                'results': ['v5'],
                'attrs': {},
            },
            {
                'name': 'lattice.sparse.with_features',
                'operands': ['v0', 'v5'],
                'results': ['v6'],
                'attrs': {},
            },
        ],
        returns=['v6'],
    )


def _activation_plan(*, kind: str) -> dict[str, object]:
    return _plan_payload(
        ops=[
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg0', 'arg1', 'arg2'],
                'results': ['v0'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
            {
                'name': 'lattice.sparse.decompose',
                'operands': ['v0'],
                'results': ['v1', 'v2', 'v3'],
                'attrs': {},
            },
            {
                'name': 'lattice.activation',
                'operands': ['v2'],
                'results': ['v4'],
                'attrs': {
                    'kind': kind,
                    'approximate': 'none',
                    'alpha': 0.01,
                    'beta': 1.0,
                    'threshold': 20.0,
                },
            },
            {
                'name': 'lattice.sparse.with_features',
                'operands': ['v0', 'v4'],
                'results': ['v5'],
                'attrs': {},
            },
        ],
        returns=['v5'],
    )


def _sparse_cat_plan() -> dict[str, object]:
    return _plan_payload(
        ops=[
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg0', 'arg1', 'arg2'],
                'results': ['v0'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
            {
                'name': 'lattice.sparse.cat',
                'operands': ['v0', 'v0'],
                'results': ['v1'],
                'attrs': {'join': 'inner'},
            },
        ],
        returns=['v1'],
    )


def _norm_plan(kind: str) -> dict[str, object]:
    weight_ops = [
        _dense_channel_weight('v4', 'norm.scale', 'channel_c'),
        _dense_channel_weight('v5', 'norm.bias', 'bias_c'),
        _dense_channel_weight('v6', 'norm.mean', 'channel_c'),
        _dense_channel_weight('v7', 'norm.var', 'channel_c'),
    ]
    norm_operands = {
        'batch_norm': ['v2', 'v4', 'v5', 'v6', 'v7'],
        'layer_norm': ['v2', 'v4', 'v5'],
        'rms_norm': ['v2', 'v4'],
    }[kind]
    return _plan_payload(
        ops=[
            {
                'name': 'lattice.sparse.make',
                'operands': ['arg0', 'arg1', 'arg2'],
                'results': ['v0'],
                'attrs': {
                    'stride': [1, 1, 1],
                    'coord_order': 'batch_x_y_z',
                },
            },
            {
                'name': 'lattice.sparse.decompose',
                'operands': ['v0'],
                'results': ['v1', 'v2', 'v3'],
                'attrs': {},
            },
            *weight_ops,
            {
                'name': f'lattice.{kind}',
                'operands': norm_operands,
                'results': ['v8'],
                'attrs': {'eps': 1e-5},
            },
            {
                'name': 'lattice.sparse.with_features',
                'operands': ['v0', 'v8'],
                'results': ['v9'],
                'attrs': {},
            },
        ],
        returns=['v9'],
    )


def _dense_channel_weight(
    result_name: str,
    storage_key: str,
    layout: str,
) -> dict[str, object]:
    return {
        'name': 'lattice.weight',
        'operands': [],
        'results': [result_name],
        'attrs': {
            'storage_key': storage_key,
            'layout': layout,
            'packing': {'kind': 'dense'},
        },
    }


def _packed_weights(prefix: str, weight) -> dict[str, mx.array]:
    return {
        f'{prefix}.weight': weight.weight,
        f'{prefix}.scales': weight.scales,
        f'{prefix}.biases': weight.biases,
    }


def _has_mlir_validator() -> bool:
    return (
        hasattr(ext, 'validate_lattice_mlir')
        or Path(
            'build/clangd-mlir/mlir/tools/lattice-opt/lattice-opt'
        ).is_file()
    )

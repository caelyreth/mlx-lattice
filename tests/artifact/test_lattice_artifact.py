from __future__ import annotations

import inspect
from typing import cast

import mlx.nn as mxnn
import pytest
from lattice_contract import (
    FEATURE_LINEAR,
    SPARSE_ADD,
    SPARSE_CONV3D,
    SPARSE_SUBM_CONV3D,
    IRNode,
    manifest_from_dict,
    manifest_to_dict,
)

from mlx_lattice import SparseTensor
from mlx_lattice import nn as lnn
from mlx_lattice.artifact import (
    LatticeArtifact,
    LatticeModel,
    load_lattice_artifact,
    load_lattice_model,
    save_lattice_graph,
    save_lattice_model,
    save_lattice_module,
)
from mlx_lattice.artifact.builder import (
    LatticeGraphBuilder,
    build_lattice_graph_artifact,
    build_lattice_module_artifact,
)
from mlx_lattice.artifact.registry import (
    iter_operation_specs,
    module_artifact_binding,
    validate_node_against_artifact,
)
from mlx_lattice.nn._artifact import module_artifact_spec
from tests.support import assert_nested_close, mx

pytestmark = [pytest.mark.usefixtures('selected_backend')]


def _input() -> SparseTensor:
    return SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
        batch_counts=(3,),
    )


def _weights() -> dict[str, mx.array]:
    return {
        'linear.weight': mx.array([[2.0], [-1.0]], dtype=mx.float32),
        'linear.bias': mx.array([1.0, 0.5], dtype=mx.float32),
        'conv.weight': mx.array(
            [
                [[1.0, 0.5]],
                [[2.0, -1.0]],
                [[0.25, 1.5]],
            ],
            dtype=mx.float32,
        ),
    }


def _assert_sparse_close(
    actual: SparseTensor, expected: SparseTensor
) -> None:
    mx.eval(actual.coords, actual.feats, expected.coords, expected.feats)
    assert actual.stride == expected.stride
    assert actual.coords.tolist() == expected.coords.tolist()
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_lattice_model_runs_semantic_feature_linear() -> None:
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
            'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
            'nodes': [
                {
                    'id': 'linear',
                    'op': FEATURE_LINEAR.name,
                    'inputs': {'input': 'input'},
                    'outputs': {'output': 'output'},
                    'parameters': {
                        'weight': 'linear.weight',
                        'bias': 'linear.bias',
                    },
                }
            ],
        }
    )
    model = LatticeModel(manifest, _weights())
    actual = model(_input())
    expected = lnn.Linear(1, 2)(_input())
    expected = expected.replace(
        feats=_input().feats @ _weights()['linear.weight'].T
        + _weights()['linear.bias']
    )
    _assert_sparse_close(actual, expected)


def test_lattice_model_runs_target_conv_support() -> None:
    x = _input()
    target = SparseTensor(
        mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32),
        mx.ones((2, 1), dtype=mx.float32),
    )
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [
                {'name': 'input', 'type': 'sparse_tensor'},
                {'name': 'target', 'type': 'sparse_tensor'},
            ],
            'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
            'nodes': [
                {
                    'id': 'conv',
                    'op': SPARSE_CONV3D.name,
                    'inputs': {'input': 'input'},
                    'outputs': {'output': 'output'},
                    'parameters': {'weight': 'conv.weight'},
                    'support': {
                        'kind': 'target',
                        'target': 'target',
                        'kernel_size': [3, 1, 1],
                    },
                }
            ],
        }
    )
    actual = LatticeModel(manifest, _weights())(x, target)
    expected = lnn.Conv3d(1, 2, kernel_size=(3, 1, 1), bias=False)
    expected.weight = _weights()['conv.weight']
    _assert_sparse_close(actual, expected(x, coordinates=target))


def test_lattice_artifact_roundtrips_through_safetensors(tmp_path) -> None:
    dense = lnn.Linear(1, 2)
    dense.weight = _weights()['linear.weight']
    dense.bias = _weights()['linear.bias']
    artifact = build_lattice_module_artifact(dense)

    save_lattice_model(tmp_path, artifact.manifest, artifact.weights)
    loaded = load_lattice_artifact(tmp_path)
    model = load_lattice_model(tmp_path)

    assert isinstance(loaded, LatticeArtifact)
    assert isinstance(loaded.model(), LatticeModel)
    _assert_sparse_close(
        model(_input()),
        LatticeModel(artifact.manifest, artifact.weights)(_input()),
    )


def test_lattice_runtime_rejects_incompatible_runtime_metadata() -> None:
    raw = manifest_to_dict(
        build_lattice_module_artifact(lnn.ReLU()).manifest
    )
    raw['runtime'] = {'name': 'other-runtime', 'version': '>=0.2.2,<0.3'}
    manifest = manifest_from_dict(raw)
    with pytest.raises(ValueError, match=r'runtime\.name'):
        LatticeModel(manifest, {})


def test_lattice_runtime_registry_is_semantic_only() -> None:
    names = {spec.name for spec in iter_operation_specs()}

    assert SPARSE_CONV3D.name in names
    assert SPARSE_SUBM_CONV3D.name in names
    assert SPARSE_ADD.name in names
    assert FEATURE_LINEAR.name in names
    assert 'pool.avg3d' in names
    assert 'pool.global_avg' in names
    assert not any(name.startswith('ops.') for name in names)
    assert not any('quantized_' in name for name in names)


def test_lattice_runtime_registry_rejects_non_canonical_attrs() -> None:
    validate_node_against_artifact(
        IRNode(
            id='subm',
            op=SPARSE_SUBM_CONV3D.name,
            inputs={'input': 'input'},
            outputs={'output': 'subm'},
            parameters={'weight': 'subm.weight'},
            attributes={'kernel_size': [1, 1, 1], 'dilation': [1, 1, 1]},
        )
    )

    with pytest.raises(ValueError, match='unsupported keys'):
        validate_node_against_artifact(
            IRNode(
                id='subm',
                op=SPARSE_SUBM_CONV3D.name,
                inputs={'input': 'input'},
                outputs={'output': 'subm'},
                parameters={'weight': 'subm.weight'},
                attributes={
                    'kernel_size': [1, 1, 1],
                    'dilation': [1, 1, 1],
                    'stride': [1, 1, 1],
                },
            )
        )


def test_lattice_module_artifact_registry_tracks_semantic_modules() -> None:
    modules = {
        name: getattr(lnn, name)
        for name in lnn.__all__
        if inspect.isclass(getattr(lnn, name))
    }
    expected_ops = {
        'AvgPool3d': 'pool.avg3d',
        'BatchNorm': 'feature.batch_norm',
        'Conv3d': 'sparse.conv3d',
        'ConvTranspose3d': 'sparse.conv_transpose3d',
        'Dropout': 'feature.dropout',
        'GELU': 'feature.gelu',
        'GenerativeConvTranspose3d': 'sparse.generative_conv_transpose3d',
        'GlobalAvgPool': 'pool.global_avg',
        'GlobalMaxPool': 'pool.global_max',
        'GlobalSumPool': 'pool.global_sum',
        'LayerNorm': 'feature.layer_norm',
        'LeakyReLU': 'feature.leaky_relu',
        'Linear': 'feature.linear',
        'MaxPool3d': 'pool.max3d',
        'Pool3d': 'pool.pool3d',
        'QuantizedConv3d': 'sparse.conv3d',
        'QuantizedConvTranspose3d': 'sparse.conv_transpose3d',
        'QuantizedGenerativeConvTranspose3d': (
            'sparse.generative_conv_transpose3d'
        ),
        'QuantizedLinear': 'feature.linear',
        'QuantizedSubmConv3d': 'sparse.subm_conv3d',
        'RMSNorm': 'feature.rms_norm',
        'ReLU': 'feature.relu',
        'SiLU': 'feature.silu',
        'Sigmoid': 'feature.sigmoid',
        'Softplus': 'feature.softplus',
        'SubmConv3d': 'sparse.subm_conv3d',
        'SumPool3d': 'pool.sum3d',
        'Tanh': 'feature.tanh',
    }

    assert set(modules) == set(expected_ops)
    for name, cls in modules.items():
        assert module_artifact_spec(cls) is not None
        instance = _module_instance(name)
        assert isinstance(instance, cls)
        assert module_artifact_binding(instance).op == expected_ops[name]


def _module_instance(name: str) -> mxnn.Module:
    if name in {
        'Conv3d',
        'SubmConv3d',
        'ConvTranspose3d',
        'GenerativeConvTranspose3d',
        'QuantizedConv3d',
        'QuantizedSubmConv3d',
        'QuantizedConvTranspose3d',
        'QuantizedGenerativeConvTranspose3d',
    }:
        return getattr(lnn, name)(1, 1)
    if name in {'Linear', 'QuantizedLinear'}:
        return getattr(lnn, name)(1, 1)
    if name in {'BatchNorm', 'LayerNorm', 'RMSNorm'}:
        return getattr(lnn, name)(1)
    return getattr(lnn, name)()


@pytest.mark.parametrize('bits', [4, 8])
def test_lattice_module_artifact_preserves_quantized_linear_storage(
    bits: int,
) -> None:
    dense = lnn.Linear(1, 2)
    dense.weight = _weights()['linear.weight']
    dense.bias = _weights()['linear.bias']
    artifact = build_lattice_module_artifact(dense.to_quantized(bits=bits))

    assert artifact.manifest.nodes[0].op == FEATURE_LINEAR.name
    prefix = artifact.manifest.nodes[0].parameters['weight']
    assert artifact.weights[f'{prefix}.weight'].dtype == mx.uint32
    attrs = cast('list[int]', artifact.weights[f'{prefix}.attrs'].tolist())
    assert int(attrs[2]) == bits
    _assert_sparse_close(
        LatticeModel(artifact.manifest, artifact.weights)(_input()),
        dense.to_quantized(bits=bits)(_input()),
    )


def test_lattice_model_rejects_incomplete_quantized_weight_payload() -> (
    None
):
    artifact = build_lattice_module_artifact(
        lnn.Linear(1, 2).to_quantized()
    )
    prefix = artifact.manifest.nodes[0].parameters['weight']
    weights = dict(artifact.weights)
    del weights[f'{prefix}.attrs']

    with pytest.raises(ValueError, match='missing quantized weight'):
        LatticeModel(artifact.manifest, weights)


def test_explicit_legacy_builder_uses_semantic_contracts() -> None:
    weight = mx.array([[2.0]], dtype=mx.float32)
    builder = LatticeGraphBuilder()
    projected = builder.call(
        FEATURE_LINEAR,
        input='input',
        weight=weight,
    )
    out = builder.add_op(
        'skip',
        SPARSE_ADD,
        inputs={'lhs': 'input', 'rhs': projected},
        attributes={'join': 'outer'},
    )
    artifact = build_lattice_graph_artifact(builder, outputs=[out])
    actual = LatticeModel(artifact.manifest, artifact.weights)(_input())

    linear = _input().replace(feats=_input().feats @ weight.T)
    expected = _input().replace(feats=_input().feats + linear.feats)
    _assert_sparse_close(actual, expected)


def test_save_lattice_module_and_graph_helpers(tmp_path) -> None:
    module_path = tmp_path / 'module'
    graph_path = tmp_path / 'graph'

    dense = lnn.Linear(1, 2)
    dense.weight = _weights()['linear.weight']
    dense.bias = _weights()['linear.bias']
    save_lattice_module(module_path, dense)

    builder = LatticeGraphBuilder()
    out = builder.call(
        FEATURE_LINEAR,
        input='input',
        weight=_weights()['linear.weight'],
        bias=_weights()['linear.bias'],
    )
    save_lattice_graph(graph_path, builder, outputs=[out])

    assert isinstance(load_lattice_model(module_path), LatticeModel)
    assert isinstance(load_lattice_model(graph_path), LatticeModel)

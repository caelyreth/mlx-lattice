from __future__ import annotations

import pytest

from lattice_contract import (
    ir_value_type,
    is_ir_value_type,
    manifest_from_dict,
    manifest_to_dict,
)


def _manifest() -> dict:
    return {
        'schema_version': '0.1',
        'producer': {'name': 'test'},
        'runtime': {'name': 'mlx-lattice', 'version': '>=0.2.1,<0.3'},
        'coordinate_order': ['batch', 'x', 'y', 'z'],
        'feature_layout': ['row', 'channel'],
        'weight_layout': 'mlx-lattice',
        'dtype_policy': 'preserve',
        'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
        'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
        'nodes': [
            {
                'id': 'relu',
                'op': 'feature.relu',
                'inputs': {'input': 'input'},
                'outputs': {'output': 'output'},
            }
        ],
    }


def test_manifest_roundtrip_preserves_semantic_contract() -> None:
    manifest = manifest_from_dict(_manifest())

    raw = manifest_to_dict(manifest)

    assert raw['schema_version'] == '0.1'
    assert raw['coordinate_order'] == ['batch', 'x', 'y', 'z']
    assert raw['feature_layout'] == ['row', 'channel']
    assert raw['weight_layout'] == 'mlx-lattice'
    assert raw['nodes'][0]['op'] == 'feature.relu'


def test_manifest_rejects_unknown_schema_version() -> None:
    raw = _manifest()
    raw['schema_version'] = '9.9'

    with pytest.raises(ValueError, match='unsupported lattice IR'):
        manifest_from_dict(raw)


def test_ir_value_type_helper_validates_schema_values() -> None:
    assert is_ir_value_type('sparse_tensor')
    assert ir_value_type('dense_tensor') == 'dense_tensor'
    assert not is_ir_value_type('not_a_value_type')

    with pytest.raises(ValueError, match='unsupported IR value type'):
        ir_value_type('not_a_value_type')


def test_manifest_rejects_unknown_graph_input_reference() -> None:
    raw = _manifest()
    raw['nodes'][0]['inputs']['input'] = 'missing'

    with pytest.raises(ValueError, match='unknown input'):
        manifest_from_dict(raw)


def test_manifest_rejects_duplicate_graph_values() -> None:
    raw = _manifest()
    raw['nodes'][0]['outputs']['output'] = 'input'

    with pytest.raises(ValueError, match='duplicate graph value'):
        manifest_from_dict(raw)

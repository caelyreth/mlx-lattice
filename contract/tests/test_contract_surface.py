from __future__ import annotations

import importlib
import sys

import pytest

from lattice_contract import (
    FEATURE_LINEAR,
    SPARSE_SUBM_CONV3D,
    IRNode,
    iter_op_contracts,
    op_contract,
    validate_node_against_spec,
)


def test_lattice_contract_imports_without_mlx_lattice_runtime() -> None:
    sys.modules.pop('lattice_contract', None)
    sys.modules.pop('lattice_contract.manifest', None)
    sys.modules.pop('lattice_contract.ops', None)
    sys.modules.pop('mlx_lattice', None)

    contract = importlib.import_module('lattice_contract')

    assert contract.__version__ == '0.2.1'
    assert contract.CURRENT_SCHEMA_VERSION == '0.1'
    assert 'mlx_lattice' not in sys.modules


def test_builtin_semantic_op_contracts_are_registered() -> None:
    contracts = {
        contract.name: contract for contract in iter_op_contracts()
    }

    assert op_contract(FEATURE_LINEAR.name) is FEATURE_LINEAR
    assert contracts[FEATURE_LINEAR.name].spec.input_types == {
        'input': 'any'
    }
    assert contracts[FEATURE_LINEAR.name].spec.output_types == {
        'output': 'any'
    }
    assert SPARSE_SUBM_CONV3D.spec.attributes == frozenset(
        {'kernel_size', 'dilation'}
    )


def test_builtin_semantic_op_contract_validates_attributes() -> None:
    validate_node_against_spec(
        IRNode(
            id='subm',
            op=SPARSE_SUBM_CONV3D.name,
            inputs={'input': 'input'},
            outputs={'output': 'output'},
            parameters={'weight': 'weight'},
            attributes={'kernel_size': [3, 3, 3], 'dilation': [1, 1, 1]},
        )
    )

    with pytest.raises(ValueError, match='unsupported keys'):
        validate_node_against_spec(
            IRNode(
                id='subm',
                op=SPARSE_SUBM_CONV3D.name,
                inputs={'input': 'input'},
                outputs={'output': 'output'},
                parameters={'weight': 'weight'},
                attributes={'unexpected': True},
            )
        )

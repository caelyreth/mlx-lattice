from __future__ import annotations

import importlib
import sys


def test_lattice_contract_imports_without_mlx_lattice_runtime() -> None:
    sys.modules.pop('lattice_contract', None)
    sys.modules.pop('mlx_lattice', None)

    contract = importlib.import_module('lattice_contract')

    assert contract.ARTIFACT_GRAPH_FILE == 'graph.mlir'
    assert contract.ARTIFACT_WEIGHT_FILE == 'weights.safetensors'
    assert contract.CURRENT_DIALECT_VERSION == 0
    assert contract.LATTICE_DIALECT.namespace == 'lattice'
    assert contract.__version__ == '0.2.2'
    assert 'mlx_lattice' not in sys.modules

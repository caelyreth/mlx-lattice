from __future__ import annotations

from lattice_contract.dialect import LATTICE_DIALECT
from lattice_contract.schema import schema_fingerprint

CURRENT_DIALECT_VERSION = 2
DIALECT_SCHEMA_DIGEST = schema_fingerprint(LATTICE_DIALECT)
ARTIFACT_GRAPH_FILE = 'graph.mlir'
ARTIFACT_WEIGHT_FILE = 'weights.safetensors'

__all__ = [
    'ARTIFACT_GRAPH_FILE',
    'ARTIFACT_WEIGHT_FILE',
    'CURRENT_DIALECT_VERSION',
    'DIALECT_SCHEMA_DIGEST',
]

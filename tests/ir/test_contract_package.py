from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

from lattice_contract import LATTICE_DIALECT


def test_lattice_contract_imports_without_mlx_lattice_runtime() -> None:
    sys.modules.pop('lattice_contract', None)
    sys.modules.pop('mlx_lattice', None)

    contract = importlib.import_module('lattice_contract')

    assert contract.ARTIFACT_GRAPH_FILE == 'graph.mlir'
    assert contract.ARTIFACT_WEIGHT_FILE == 'weights.safetensors'
    assert contract.CURRENT_DIALECT_VERSION == 0
    assert contract.LATTICE_DIALECT.namespace == 'lattice'
    assert contract.__version__ == '0.3.1'
    assert 'mlx_lattice' not in sys.modules


def test_python_contract_matches_committed_lattice_ods_surface() -> None:
    root = Path('mlir/include/Lattice/Dialect/Lattice/IR')
    types_td = (root / 'LatticeTypes.td').read_text(encoding='utf-8')
    attrs_td = (root / 'LatticeAttrs.td').read_text(encoding='utf-8')
    ops_td = (root / 'LatticeOps.td').read_text(encoding='utf-8')

    assert _td_definitions(types_td, 'Lattice_Type') == tuple(
        (item.name, item.mnemonic)
        for item in LATTICE_DIALECT.types.values()
    )
    assert _td_definitions(attrs_td, 'Lattice_Attr') == tuple(
        (item.name, item.mnemonic)
        for item in LATTICE_DIALECT.attrs.values()
    )
    assert _td_ops(ops_td) == tuple(
        f'lattice.{item.name}' for item in LATTICE_DIALECT.ops.values()
    )


def _td_definitions(
    text: str, td_class: str
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, mnemonic)
        for name, mnemonic in re.findall(
            rf'def\s+Lattice_\w+\s*:\s*{td_class}<"([^"]+)",\s*"([^"]+)">',
            text,
            flags=re.S,
        )
    )


def _td_ops(text: str) -> tuple[str, ...]:
    return tuple(
        f'lattice.{name}'
        for name in re.findall(
            r'def\s+Lattice_\w+\s*:\s*Lattice_Op<"([^"]+)"',
            text,
            flags=re.S,
        )
    )

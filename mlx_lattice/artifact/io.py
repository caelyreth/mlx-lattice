from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx
from lattice_contract import ARTIFACT_GRAPH_FILE, ARTIFACT_WEIGHT_FILE

_GRAPH_NAME = ARTIFACT_GRAPH_FILE
_WEIGHTS_NAME = ARTIFACT_WEIGHT_FILE


@dataclass(frozen=True, slots=True)
class LatticeArtifact:
    """MLIR-first lattice artifact bundle.

    The artifact is intentionally just the exchange media at this layer:
    textual lattice MLIR plus safetensors weights. Executable MLX import is a
    separate lowering step built on top of the MLIR contract.
    """

    graph: str
    weights: dict[str, mx.array]


def save_lattice_artifact(
    path: str | Path,
    graph: str,
    weights: dict[str, mx.array] | None = None,
) -> None:
    """Write a lattice MLIR artifact directory."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / _GRAPH_NAME).write_text(graph, encoding='utf-8')
    mx.save_safetensors(str(root / _WEIGHTS_NAME), dict(weights or {}))


def load_lattice_artifact(path: str | Path) -> LatticeArtifact:
    """Load a lattice MLIR artifact directory."""

    root = Path(path)
    if not root.is_dir():
        raise ValueError(
            f'lattice artifact directory does not exist: {root}'
        )
    graph_path = root / _GRAPH_NAME
    weights_path = root / _WEIGHTS_NAME
    if not graph_path.is_file():
        raise ValueError(f'lattice artifact is missing {_GRAPH_NAME}.')
    if not weights_path.is_file():
        raise ValueError(f'lattice artifact is missing {_WEIGHTS_NAME}.')
    weights = mx.load(str(weights_path))
    if not isinstance(weights, dict):
        raise ValueError(
            'weights.safetensors must load as a tensor mapping.'
        )
    return LatticeArtifact(
        graph=graph_path.read_text(encoding='utf-8'),
        weights=cast(dict[str, mx.array], weights),
    )

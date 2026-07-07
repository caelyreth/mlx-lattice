from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mlx_lattice import _ext as ext

from .io import LatticeArtifact, load_lattice_artifact

_LOCAL_LATTICE_OPT = Path(
    'build/clangd-mlir/mlir/tools/lattice-opt/lattice-opt'
)


@dataclass(frozen=True, slots=True)
class LatticeMLIRStatus:
    """Native MLIR validation result for a lattice graph."""

    valid: bool
    diagnostics: str = ''


def validate_lattice_graph(
    graph: str,
    *,
    lattice_opt: str | Path | None = None,
) -> None:
    """Parse and verify lattice MLIR with native MLIR infrastructure.

    MLIR-enabled native builds use the in-process parser/verifier binding.
    Lightweight builds can still validate through ``lattice-opt`` by passing a
    tool path, setting ``MLX_LATTICE_LATTICE_OPT``, or building the repository
    ``clangd-mlir`` preset.
    """

    native = getattr(ext, 'validate_lattice_mlir', None)
    if native is not None:
        native(graph)
        return
    _validate_with_lattice_opt(graph, _resolve_lattice_opt(lattice_opt))


def lattice_graph_status(
    graph: str,
    *,
    lattice_opt: str | Path | None = None,
) -> LatticeMLIRStatus:
    """Return MLIR parse/verify status for a lattice graph."""

    native = getattr(ext, 'lattice_mlir_status', None)
    if native is not None:
        raw = native(graph)
        return LatticeMLIRStatus(
            valid=bool(raw['valid']),
            diagnostics=str(raw['diagnostics']),
        )
    tool = _resolve_lattice_opt(lattice_opt)
    try:
        _validate_with_lattice_opt(graph, tool)
    except ValueError as exc:
        return LatticeMLIRStatus(valid=False, diagnostics=str(exc))
    return LatticeMLIRStatus(valid=True)


def lattice_graph_operation_names(graph: str) -> tuple[str, ...]:
    """Return lattice operation names from a native-verified graph.

    This requires an MLIR-enabled native extension. It is the first typed
    importer bridge and deliberately does not fall back to textual parsing.
    """

    native = getattr(ext, 'lattice_mlir_operation_names', None)
    if native is None:
        raise RuntimeError(
            'lattice operation inspection requires an MLIR-enabled '
            'mlx-lattice native extension.'
        )
    return tuple(str(item) for item in native(graph))


def validate_lattice_artifact(
    artifact: LatticeArtifact | str | Path,
    *,
    lattice_opt: str | Path | None = None,
) -> None:
    """Parse and verify the graph contained in a lattice artifact."""

    loaded = (
        load_lattice_artifact(artifact)
        if isinstance(artifact, str | Path)
        else artifact
    )
    validate_lattice_graph(loaded.graph, lattice_opt=lattice_opt)


def lattice_artifact_status(
    artifact: LatticeArtifact | str | Path,
    *,
    lattice_opt: str | Path | None = None,
) -> LatticeMLIRStatus:
    """Return MLIR validation status for a lattice artifact."""

    loaded = (
        load_lattice_artifact(artifact)
        if isinstance(artifact, str | Path)
        else artifact
    )
    return lattice_graph_status(loaded.graph, lattice_opt=lattice_opt)


def _resolve_lattice_opt(explicit: str | Path | None) -> Path:
    if explicit is not None:
        return _require_tool(Path(explicit))
    env = os.environ.get('MLX_LATTICE_LATTICE_OPT')
    if env:
        return _require_tool(Path(env))
    if _LOCAL_LATTICE_OPT.is_file():
        return _LOCAL_LATTICE_OPT
    raise RuntimeError(
        'lattice MLIR validation requires either an MLIR-enabled native '
        'extension or a lattice-opt executable. Build with '
        '`cmake --preset clangd-mlir && cmake --build --preset clangd-mlir '
        '--target lattice-opt`, pass lattice_opt=..., or set '
        'MLX_LATTICE_LATTICE_OPT.'
    )


def _require_tool(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f'lattice-opt executable does not exist: {path}')
    return path


def _validate_with_lattice_opt(graph: str, tool: Path) -> None:
    with tempfile.TemporaryDirectory(prefix='mlx-lattice-mlir-') as root:
        source = Path(root) / 'graph.mlir'
        output = Path(root) / 'out.mlir'
        source.write_text(graph, encoding='utf-8')
        result = subprocess.run(
            [str(tool), str(source), '-o', str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        diagnostics = result.stderr.strip() or result.stdout.strip()
        raise ValueError(diagnostics or 'lattice MLIR validation failed.')

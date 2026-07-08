from __future__ import annotations

from mlx_lattice import _ext as ext


def backend_info() -> dict[str, object]:
    """Return version and compiled native capability diagnostics.

    Capabilities include semantic CPU support, Metal backend availability, and
    native MLIR artifact execution support. They are diagnostic booleans, not
    route-selection controls.
    """

    capabilities = dict(ext.capabilities())
    capabilities.setdefault(
        'mlir',
        callable(getattr(ext, 'lattice_mlir_plan', None)),
    )
    return {
        'version': ext.version(),
        'capabilities': capabilities,
    }

from __future__ import annotations

from mlx_lattice import _ext as ext
from mlx_lattice.backends import cuda


def backend_info() -> dict[str, object]:
    capabilities = ext.capabilities()
    return {
        'version': ext.version(),
        'capabilities': capabilities,
        'cuda': cuda.info(),
    }

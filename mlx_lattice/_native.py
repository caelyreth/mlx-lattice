from __future__ import annotations

from . import _ext


def backend_info() -> dict[str, object]:
    return {
        'version': _ext.version(),
        'capabilities': _ext.capabilities(),
    }

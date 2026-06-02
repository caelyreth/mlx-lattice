from __future__ import annotations

from ._native import add_ints, backend_info

__all__ = ['__version__', 'add_ints', 'backend_info']

__version__ = backend_info()['version']

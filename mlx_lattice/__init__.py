from __future__ import annotations

from . import ops as ops
from ._native import backend_info
from .coords import (
    CoordinateManager,
    CoordinateMapKey,
    contains_coords,
    inverse_map,
    lookup_coords,
)
from .ops import cat, prune, sparse_collate, topk_rows
from .tensor import SparseTensor

__all__ = [
    'CoordinateManager',
    'CoordinateMapKey',
    'SparseTensor',
    '__version__',
    'backend_info',
    'cat',
    'contains_coords',
    'inverse_map',
    'lookup_coords',
    'ops',
    'prune',
    'sparse_collate',
    'topk_rows',
]

__version__ = backend_info()['version']

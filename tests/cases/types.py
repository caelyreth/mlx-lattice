from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValueCase:
    name: str
    run: Callable[[], object]
    abs: float = 1e-6
    marks: tuple[Any, ...] = ()

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mlx_lattice.core.types import Triple, triple


@dataclass(frozen=True, slots=True, init=False)
class KernelSpec:
    size: Triple
    stride: Triple
    padding: Triple
    dilation: Triple

    def __init__(
        self,
        size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        normalized_size = triple(size, name='kernel_size')
        normalized_stride = triple(stride, name='stride')
        normalized_padding = triple(padding, name='padding')
        normalized_dilation = triple(dilation, name='dilation')
        _require_positive(normalized_size, 'kernel_size')
        _require_positive(normalized_stride, 'stride')
        _require_nonnegative(normalized_padding, 'padding')
        _require_positive(normalized_dilation, 'dilation')

        object.__setattr__(self, 'size', normalized_size)
        object.__setattr__(self, 'stride', normalized_stride)
        object.__setattr__(self, 'padding', normalized_padding)
        object.__setattr__(self, 'dilation', normalized_dilation)

    @property
    def volume(self) -> int:
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def is_pointwise(self) -> bool:
        return (
            self.size == (1, 1, 1)
            and self.stride == (1, 1, 1)
            and self.padding == (0, 0, 0)
            and self.dilation == (1, 1, 1)
        )

    @property
    def is_centered_submanifold(self) -> bool:
        return (
            self.stride == (1, 1, 1)
            and self.padding == (0, 0, 0)
            and self.dilation == (1, 1, 1)
            and all(value % 2 == 1 for value in self.size)
        )


# MARK: - helpers


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')


def _require_nonnegative(values: Triple, name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f'{name} values must be non-negative.')

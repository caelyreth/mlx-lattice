from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, overload

import mlx.core as mx
from lattice_contract import LATTICE_DIALECT, OpDef

from mlx_lattice.core import SparseTensor

type RuntimeValue = SparseTensor | mx.array
type SparseJoin = Literal['inner', 'left', 'right', 'outer']
type LoweringResult = RuntimeValue | tuple[RuntimeValue, ...]
type LatticeOpHandle = str | OpDef | Callable[..., Any]


class LoweringProgram(Protocol):
    """Program context required by artifact lowerings."""

    weights: Mapping[str, mx.array]


class LoweringFn(Protocol):
    """Runtime lowering callable with a stable function name."""

    __name__: str

    def __call__(
        self,
        program: LoweringProgram,
        operation: Mapping[str, Any],
        values: dict[str, RuntimeValue],
    ) -> LoweringResult: ...


@dataclass(slots=True)
class LoweringRegistry:
    """Annotation-backed MLX lowering registry for lattice MLIR ops."""

    functions: dict[str, LoweringFn]

    def register(self, op: LatticeOpHandle, fn: LoweringFn) -> LoweringFn:
        """Register ``fn`` as the MLX artifact lowering for ``op``."""

        definition = LATTICE_DIALECT.resolve_op(op)
        name = f'{LATTICE_DIALECT.namespace}.{definition.name}'
        if name in self.functions:
            raise ValueError(f'duplicate artifact lowering: {name}')
        self.functions[name] = fn
        return fn

    def lower(
        self,
        program: LoweringProgram,
        operation: Mapping[str, Any],
        values: dict[str, RuntimeValue],
    ) -> LoweringResult:
        """Run the registered lowering for one native MLIR plan operation."""

        name = str(operation['name'])
        try:
            lowering = self.functions[name]
        except KeyError as exc:
            raise ValueError(
                f'unsupported artifact lowering op: {name}'
            ) from exc
        return lowering(program, operation, values)


ARTIFACT_LOWERINGS = LoweringRegistry({})


@overload
def lattice_lowering[OpFnT: Callable[..., Any]](fn: OpFnT, /) -> OpFnT: ...


@overload
def lattice_lowering[OpFnT: Callable[..., Any]](
    fn: None = None, /, *, op: LatticeOpHandle
) -> Callable[[OpFnT], OpFnT]: ...


def lattice_lowering[OpFnT: Callable[..., Any]](
    fn: OpFnT | None = None,
    /,
    *,
    op: LatticeOpHandle | None = None,
) -> OpFnT | Callable[[OpFnT], OpFnT]:
    """Attach lattice dialect metadata to a framework API function."""

    def decorator(api: OpFnT) -> OpFnT:
        definition = LATTICE_DIALECT.resolve_op(op or _callable_name(api))
        api.__dict__['__lattice_op__'] = definition
        return api

    if fn is None:
        return decorator
    return decorator(fn)


def artifact_lowering(
    *,
    op: LatticeOpHandle,
    dialect_op: LatticeOpHandle | None = None,
) -> Callable[[LoweringFn], LoweringFn]:
    """Bind an artifact lowering to a lattice op or API declaration."""

    target = dialect_op or op

    def decorator(fn: LoweringFn) -> LoweringFn:
        return ARTIFACT_LOWERINGS.register(target, fn)

    return decorator


def _callable_name(fn: Callable[..., Any]) -> str:
    name = cast(Any, fn).__name__
    if not isinstance(name, str):
        raise TypeError('lattice lowering declarations must be named.')
    return name


def operands(operation: Mapping[str, Any]) -> tuple[str, ...]:
    """Return operation operand value names."""

    return tuple(
        str(item) for item in cast(Sequence[Any], operation['operands'])
    )


def results(operation: Mapping[str, Any]) -> tuple[str, ...]:
    """Return operation result value names."""

    return tuple(
        str(item) for item in cast(Sequence[Any], operation['results'])
    )


def attrs(operation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return operation attributes."""

    return cast(Mapping[str, Any], operation['attrs'])


def sparse(values: Mapping[str, RuntimeValue], name: str) -> SparseTensor:
    """Read a sparse value from the runtime environment."""

    value = values[name]
    if not isinstance(value, SparseTensor):
        raise TypeError(f'{name} must be a SparseTensor.')
    return value


def array(values: Mapping[str, RuntimeValue], name: str) -> mx.array:
    """Read an MLX array from the runtime environment."""

    value = values[name]
    if isinstance(value, SparseTensor):
        raise TypeError(f'{name} must be an MLX array.')
    return value


def str_attr(attrs: Mapping[str, Any], name: str) -> str:
    """Read a string attribute."""

    value = attrs[name]
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string attribute.')
    return value


def float_attr(attrs: Mapping[str, Any], name: str) -> float:
    """Read a floating-point attribute."""

    value = attrs[name]
    if not isinstance(value, int | float):
        raise TypeError(f'{name} must be numeric.')
    return float(value)


def triple_attr(
    attrs: Mapping[str, Any], name: str
) -> tuple[int, int, int]:
    """Read an integer triple attribute."""

    value = attrs[name]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f'{name} must be an integer triple.')
    items = tuple(int(item) for item in value)
    if len(items) != 3:
        raise ValueError(f'{name} must contain exactly 3 integers.')
    return cast(tuple[int, int, int], items)


def join_attr(attrs: Mapping[str, Any], name: str) -> SparseJoin:
    """Read a sparse join attribute."""

    value = str_attr(attrs, name)
    if value not in ('inner', 'left', 'right', 'outer'):
        raise ValueError(f'unsupported sparse join mode: {value}')
    return cast(SparseJoin, value)

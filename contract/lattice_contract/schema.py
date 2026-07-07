from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Self, TypeVar

DeclarationT = TypeVar('DeclarationT', bound=Callable | type)

ValueKind = Literal['ssa', 'symbol']
AttributeKind = Literal[
    'coord',
    'feature_layout',
    'weight_layout',
    'packing',
    'join',
    'i64_triple',
    'f32',
    'str',
]


@dataclass(frozen=True, slots=True)
class TypeParameter:
    """One MLIR dialect type parameter."""

    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class TypeDef:
    """Annotated MLIR type declaration."""

    name: str
    mnemonic: str
    parameters: tuple[TypeParameter, ...] = ()
    summary: str = ''


@dataclass(frozen=True, slots=True)
class AttrParameter:
    """One MLIR dialect attribute parameter."""

    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class AttrDef:
    """Annotated MLIR attribute declaration."""

    name: str
    mnemonic: str
    parameters: tuple[AttrParameter, ...] = ()
    values: tuple[str, ...] = ()
    summary: str = ''


@dataclass(frozen=True, slots=True)
class OperandDef:
    """One operation operand declaration."""

    name: str
    type: str
    kind: ValueKind = 'ssa'


@dataclass(frozen=True, slots=True)
class ResultDef:
    """One operation result declaration."""

    name: str
    type: str


@dataclass(frozen=True, slots=True)
class OpAttributeDef:
    """One operation attribute declaration."""

    name: str
    kind: AttributeKind
    required: bool = True


@dataclass(frozen=True, slots=True)
class OpDef:
    """Annotated MLIR operation declaration."""

    name: str
    python_name: str
    operands: tuple[OperandDef, ...] = ()
    results: tuple[ResultDef, ...] = ()
    attributes: tuple[OpAttributeDef, ...] = ()
    assembly: str = 'functional'
    summary: str = ''


@dataclass(slots=True)
class DialectSchema:
    """Annotation-backed dialect schema.

    The schema is intentionally executable: decorators register MLIR
    declarations once, and downstream builders/importers consume the registry
    instead of maintaining hand-written op-name maps.
    """

    namespace: str
    types: dict[str, TypeDef] = field(default_factory=dict)
    attrs: dict[str, AttrDef] = field(default_factory=dict)
    ops: dict[str, OpDef] = field(default_factory=dict)
    _ops_by_python_name: dict[str, OpDef] = field(default_factory=dict)

    def type(
        self,
        name: str,
        mnemonic: str,
        *,
        parameters: Iterable[TypeParameter] = (),
        summary: str = '',
    ) -> Callable[[DeclarationT], DeclarationT]:
        """Register a dialect type declaration."""

        def decorator(declaration: DeclarationT) -> DeclarationT:
            self.add_type(
                TypeDef(
                    name=name,
                    mnemonic=mnemonic,
                    parameters=tuple(parameters),
                    summary=summary,
                )
            )
            return declaration

        return decorator

    def attr(
        self,
        name: str,
        mnemonic: str,
        *,
        parameters: Iterable[AttrParameter] = (),
        values: Iterable[str] = (),
        summary: str = '',
    ) -> Callable[[DeclarationT], DeclarationT]:
        """Register a dialect attribute declaration."""

        def decorator(declaration: DeclarationT) -> DeclarationT:
            self.add_attr(
                AttrDef(
                    name=name,
                    mnemonic=mnemonic,
                    parameters=tuple(parameters),
                    values=tuple(values),
                    summary=summary,
                )
            )
            return declaration

        return decorator

    def op(
        self,
        name: str,
        *,
        python_name: str | None = None,
        operands: Iterable[OperandDef] = (),
        results: Iterable[ResultDef] = (),
        attributes: Iterable[OpAttributeDef] = (),
        assembly: str = 'functional',
        summary: str = '',
    ) -> Callable[[DeclarationT], DeclarationT]:
        """Register a dialect operation declaration."""

        def decorator(declaration: DeclarationT) -> DeclarationT:
            self.add_op(
                OpDef(
                    name=name,
                    python_name=python_name or name.replace('.', '_'),
                    operands=tuple(operands),
                    results=tuple(results),
                    attributes=tuple(attributes),
                    assembly=assembly,
                    summary=summary,
                )
            )
            return declaration

        return decorator

    def add_type(self, definition: TypeDef) -> Self:
        if definition.name in self.types:
            raise ValueError(f'duplicate MLIR type: {definition.name}')
        self.types[definition.name] = definition
        return self

    def add_attr(self, definition: AttrDef) -> Self:
        if definition.name in self.attrs:
            raise ValueError(f'duplicate MLIR attr: {definition.name}')
        self.attrs[definition.name] = definition
        return self

    def add_op(self, definition: OpDef) -> Self:
        if definition.name in self.ops:
            raise ValueError(f'duplicate MLIR op: {definition.name}')
        if definition.python_name in self._ops_by_python_name:
            raise ValueError(
                f'duplicate MLIR op python name: {definition.python_name}'
            )
        self.ops[definition.name] = definition
        self._ops_by_python_name[definition.python_name] = definition
        return self

    def op_by_python_name(self, name: str) -> OpDef:
        try:
            return self._ops_by_python_name[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def iter_ops(self) -> Iterator[OpDef]:
        return iter(self.ops.values())


def type_param(name: str, kind: str) -> TypeParameter:
    """Declare one dialect type parameter."""

    return TypeParameter(name, kind)


def attr_param(name: str, kind: str) -> AttrParameter:
    """Declare one dialect attribute parameter."""

    return AttrParameter(name, kind)


def operand(name: str, type: str, *, kind: ValueKind = 'ssa') -> OperandDef:
    """Declare one operation operand."""

    return OperandDef(name, type, kind)


def result(name: str, type: str) -> ResultDef:
    """Declare one operation result."""

    return ResultDef(name, type)


def op_attr(
    name: str,
    kind: AttributeKind,
    *,
    required: bool = True,
) -> OpAttributeDef:
    """Declare one operation attribute."""

    return OpAttributeDef(name, kind, required)


def schema_digest(schema: DialectSchema) -> Mapping[str, tuple[str, ...]]:
    """Return a compact stable digest useful for drift tests."""

    return {
        'types': tuple(schema.types),
        'attrs': tuple(schema.attrs),
        'ops': tuple(schema.ops),
    }

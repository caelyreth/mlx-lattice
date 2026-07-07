from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self, TypeVar

DeclarationT = TypeVar('DeclarationT', bound=Callable | type)

ValueKind = Literal['ssa', 'symbol']
AttributeKind = Literal[
    'coord',
    'feature_layout',
    'weight_layout',
    'packing',
    'activation',
    'gelu_approx',
    'join',
    'binary_op',
    'pool_mode',
    'voxel_reduction',
    'point_interpolation',
    'i64_triple',
    'f64_triple',
    'i64',
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
    optional: bool = False


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
            definition = OpDef(
                name=name,
                python_name=python_name or name.replace('.', '_'),
                operands=tuple(operands),
                results=tuple(results),
                attributes=tuple(attributes),
                assembly=assembly,
                summary=summary,
            )
            self.add_op(definition)
            declaration.__dict__['__lattice_op__'] = definition
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

    def qualified_op_name(self, definition: str | OpDef) -> str:
        """Return the fully-qualified MLIR operation name."""

        op = self.resolve_op(definition)
        return f'{self.namespace}.{op.name}'

    def resolve_qualified_op(self, name: str) -> OpDef:
        """Resolve a fully-qualified MLIR operation name."""

        prefix = f'{self.namespace}.'
        if not name.startswith(prefix):
            raise ValueError(
                f'operation name must use {self.namespace!r} namespace: '
                f'{name}'
            )
        try:
            return self.ops[name.removeprefix(prefix)]
        except KeyError as exc:
            raise ValueError(
                f'unknown {self.namespace} operation: {name}'
            ) from exc

    def resolve_op(self, value: str | OpDef | Callable[..., Any]) -> OpDef:
        """Resolve an op handle to its schema definition.

        Decorated functions/classes carry ``__lattice_op__`` metadata. This is
        the preferred path because it lets framework code bind to the API
        object instead of repeating dialect strings.
        """

        if isinstance(value, OpDef):
            return value
        if isinstance(value, str):
            if value.startswith(f'{self.namespace}.'):
                return self.resolve_qualified_op(value)
            try:
                return self.ops[value]
            except KeyError:
                return self.op_by_python_name(value)
        op = getattr(value, '__lattice_op__', None)
        if isinstance(op, OpDef):
            return op
        raise TypeError(
            'expected an OpDef, op name, python op name, or declaration '
            'annotated with __lattice_op__.'
        )

    def iter_ops(self) -> Iterator[OpDef]:
        return iter(self.ops.values())


def type_param(name: str, kind: str) -> TypeParameter:
    """Declare one dialect type parameter."""

    return TypeParameter(name, kind)


def attr_param(name: str, kind: str) -> AttrParameter:
    """Declare one dialect attribute parameter."""

    return AttrParameter(name, kind)


def operand(
    name: str,
    type: str,
    *,
    kind: ValueKind = 'ssa',
    optional: bool = False,
) -> OperandDef:
    """Declare one operation operand."""

    return OperandDef(name, type, kind, optional)


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


def canonical_schema(schema: DialectSchema) -> Mapping[str, Any]:
    """Return the canonical public schema used for artifact compatibility."""

    return {
        'namespace': schema.namespace,
        'types': tuple(
            {
                'name': item.name,
                'mnemonic': item.mnemonic,
                'parameters': tuple(
                    {'name': param.name, 'kind': param.kind}
                    for param in item.parameters
                ),
            }
            for item in schema.types.values()
        ),
        'attrs': tuple(
            {
                'name': item.name,
                'mnemonic': item.mnemonic,
                'parameters': tuple(
                    {'name': param.name, 'kind': param.kind}
                    for param in item.parameters
                ),
                'values': item.values,
            }
            for item in schema.attrs.values()
        ),
        'ops': tuple(
            {
                'name': item.name,
                'python_name': item.python_name,
                'operands': tuple(
                    {
                        'name': operand.name,
                        'type': operand.type,
                        'kind': operand.kind,
                        'optional': operand.optional,
                    }
                    for operand in item.operands
                ),
                'results': tuple(
                    {'name': result.name, 'type': result.type}
                    for result in item.results
                ),
                'attributes': tuple(
                    {
                        'name': attr.name,
                        'kind': attr.kind,
                        'required': attr.required,
                    }
                    for attr in item.attributes
                ),
                'assembly': item.assembly,
            }
            for item in schema.ops.values()
        ),
    }


def schema_fingerprint(schema: DialectSchema) -> str:
    """Return a deterministic SHA-256 fingerprint for a dialect schema."""

    payload = json.dumps(
        canonical_schema(schema),
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from lattice_contract.artifact import (
    ARTIFACT_WEIGHT_FILE,
    CURRENT_DIALECT_VERSION,
    DIALECT_SCHEMA_DIGEST,
)
from lattice_contract.dialect import (
    LATTICE_DIALECT,
    sparse_decompose,
    weight,
)
from lattice_contract.schema import OpDef

Triple = tuple[int, int, int]
InputRole = Literal[
    'tensor',
    'sparse_coords',
    'sparse_features',
    'sparse_active',
]
OutputRole = Literal['tensor', 'sparse_tensor']


@dataclass(frozen=True, slots=True)
class TensorType:
    """Raw MLIR tensor type used at graph ABI boundaries."""

    text: str

    def mlir(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class SparseTensorType:
    """Lattice sparse tensor type."""

    dtype: str = 'f16'
    rank: int = 3
    coord: str = 'batch_x_y_z'
    feature: str = 'row_channel'

    def mlir(self) -> str:
        return (
            f'!lattice.sparse_tensor<rank = {self.rank}, '
            f'coord = {self.coord}, feature = {self.feature}, '
            f'dtype = {self.dtype}>'
        )


@dataclass(frozen=True, slots=True)
class WeightType:
    """Lattice symbolic weight type."""

    family: str
    dtype: str = 'f16'

    def mlir(self) -> str:
        return f'!lattice.weight<{self.family}, {self.dtype}>'


@dataclass(frozen=True, slots=True)
class Packing:
    """Lattice weight packing attribute."""

    kind: str = 'dense'
    group_size: int = 0
    scale_dtype: str = ''
    mode: str = ''

    def mlir(self) -> str:
        if self.kind == 'dense':
            return '#lattice.packing<dense>'
        return (
            f'#lattice.packing<{self.kind}, '
            f'group_size = {self.group_size}, '
            f'scale_dtype = {self.scale_dtype}, mode = {self.mode}>'
        )


@dataclass(frozen=True, slots=True)
class SSAValue:
    """Named SSA value plus its MLIR type."""

    name: str
    type: TensorType | SparseTensorType | WeightType

    def ref(self) -> str:
        return f'%{self.name}'


def dense_packing() -> Packing:
    """Return dense weight packing."""

    return Packing()


def quantized_packing(
    kind: str,
    *,
    group_size: int,
    scale_dtype: str = 'f16',
    mode: str = 'affine',
) -> Packing:
    """Return affine int4/int8 weight packing."""

    if kind not in {'int4', 'int8'}:
        raise ValueError("quantized packing kind must be 'int4' or 'int8'.")
    return Packing(kind, group_size, scale_dtype, mode)


type EmitResult = SSAValue | tuple[SSAValue, ...]
type OpHandle = str | OpDef | Callable[..., Any]


class EmitFn(Protocol):
    """Textual MLIR emitter callable with a stable function name."""

    __name__: str

    def __call__(
        self,
        builder: MLIRModuleBuilder,
        op: OpDef,
        kwargs: Mapping[str, Any],
    ) -> EmitResult: ...


@dataclass(slots=True)
class EmitRegistry:
    """Annotation-backed textual assembly emitter registry."""

    functions: dict[str, EmitFn]
    default: EmitFn

    def register(self, op: OpHandle, fn: EmitFn) -> EmitFn:
        """Register a specialized emitter for a dialect operation."""

        definition = LATTICE_DIALECT.resolve_op(op)
        if definition.name in self.functions:
            raise ValueError(f'duplicate MLIR emitter: {definition.name}')
        self.functions[definition.name] = fn
        return fn

    def emit(
        self,
        builder: MLIRModuleBuilder,
        op: OpDef,
        kwargs: Mapping[str, Any],
    ) -> EmitResult:
        """Emit ``op`` through a specialized or default emitter."""

        return self.functions.get(op.name, self.default)(
            builder, op, kwargs
        )


def mlir_emitter(*, op: OpHandle) -> Callable[[EmitFn], EmitFn]:
    """Bind a textual assembly hook to a dialect declaration."""

    def decorator(fn: EmitFn) -> EmitFn:
        return EMITTERS.register(op, fn)

    return decorator


class MLIRModuleBuilder:
    """Annotation-driven textual MLIR module builder.

    Operation methods are generated dynamically from
    :data:`lattice_contract.dialect.LATTICE_DIALECT`. The builder is therefore
    a convenience layer over the annotated schema, not a second op registry.
    """

    def __init__(self, name: str = 'forward') -> None:
        self.name = name
        self._args: list[SSAValue] = []
        self._arg_roles: list[str] = []
        self._ops: list[str] = []
        self._return: SSAValue | tuple[SSAValue, ...] | None = None
        self._return_names: tuple[str, ...] | None = None
        self._return_roles: tuple[str, ...] | None = None
        self._name_counts: dict[str, int] = {}

    def argument(
        self,
        name: str,
        type: str | TensorType | SparseTensorType,
        *,
        role: InputRole = 'tensor',
    ) -> SSAValue:
        """Append a function argument and return its SSA value."""

        value = SSAValue(name, _type(type))
        self._args.append(value)
        self._arg_roles.append(role)
        return value

    def return_(
        self,
        *values: SSAValue,
        names: Sequence[str] | None = None,
        roles: Sequence[OutputRole] | None = None,
    ) -> None:
        """Set function return values."""

        if not values:
            raise ValueError(
                'MLIR function must return at least one value.'
            )
        if names is not None and len(names) != len(values):
            raise ValueError('output names must match return value count.')
        if roles is not None and len(roles) != len(values):
            raise ValueError('output roles must match return value count.')
        self._return = values[0] if len(values) == 1 else values
        self._return_names = tuple(names) if names is not None else None
        self._return_roles = tuple(roles) if roles is not None else None

    def to_mlir(self) -> str:
        """Render the module as textual MLIR."""

        if self._return is None:
            raise ValueError('MLIR module has no return value.')
        returns = (
            self._return
            if isinstance(self._return, tuple)
            else (self._return,)
        )
        args = ',\n    '.join(
            f'{value.ref()}: {_mlir_type(value.type)}'
            for value in self._args
        )
        result_types = ', '.join(
            _mlir_type(value.type) for value in returns
        )
        return_names = self._return_names or tuple(
            value.name for value in returns
        )
        return_roles = self._return_roles or tuple(
            _default_output_role(value) for value in returns
        )
        body = '\n'.join(f'    {line}' for line in self._ops)
        return_values = ', '.join(value.ref() for value in returns)
        return_types = ', '.join(
            _mlir_type(value.type) for value in returns
        )
        return (
            'module attributes {\n'
            f'  lattice.ir_version = {CURRENT_DIALECT_VERSION},\n'
            f'  lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}",\n'
            f'  lattice.input_names = {_string_array(value.name for value in self._args)},\n'
            f'  lattice.input_roles = {_string_array(self._arg_roles)},\n'
            f'  lattice.output_names = {_string_array(return_names)},\n'
            f'  lattice.output_roles = {_string_array(return_roles)},\n'
            f'  lattice.weight_file = "{ARTIFACT_WEIGHT_FILE}"\n'
            '} {\n'
            f'  func.func @{self.name}(\n'
            f'    {args}\n'
            f'  ) -> {result_types} {{\n'
            f'{body}\n'
            f'    return {return_values} : {return_types}\n'
            '  }\n'
            '}\n'
        )

    def __getattr__(self, name: str) -> Callable[..., Any]:
        op = LATTICE_DIALECT.op_by_python_name(name)

        def emit_generated_op(
            **kwargs: Any,
        ) -> SSAValue | tuple[SSAValue, ...]:
            return self.emit(op, **kwargs)

        return emit_generated_op

    def emit(
        self, definition: OpDef, **kwargs: Any
    ) -> SSAValue | tuple[SSAValue, ...]:
        """Emit an operation declared by the annotated schema."""

        return EMITTERS.emit(self, definition, kwargs)

    def _result_value(
        self,
        op: OpDef,
        type: TensorType | SparseTensorType | WeightType,
        kwargs: Mapping[str, Any],
        *,
        index: int = 0,
    ) -> SSAValue:
        explicit = kwargs.get('result')
        if explicit is None and index > 0:
            explicit = kwargs.get(f'result_{index}')
        name = (
            str(explicit)
            if explicit is not None
            else self._unique(op.python_name.replace('_', '.'))
        )
        return SSAValue(name, type)

    def _unique(self, stem: str) -> str:
        normalized = stem.replace('.', '_')
        count = self._name_counts.get(normalized, 0)
        self._name_counts[normalized] = count + 1
        return normalized if count == 0 else f'{normalized}{count}'


def functional_emitter(
    builder: MLIRModuleBuilder,
    op: OpDef,
    kwargs: Mapping[str, Any],
) -> SSAValue:
    operands = [
        _require_value(kwargs, operand.name)
        for operand in op.operands
        if operand.kind == 'ssa'
        and (not operand.optional or kwargs.get(operand.name) is not None)
    ]
    result_type = _require_any(kwargs, 'result_type')
    result = builder._result_value(op, _type(result_type), kwargs)
    attrs = _format_attrs(op, kwargs)
    operand_refs = ', '.join(value.ref() for value in operands)
    operand_types = ', '.join(_mlir_type(value.type) for value in operands)
    builder._ops.append(
        f'{result.ref()} = {LATTICE_DIALECT.qualified_op_name(op)} '
        f'{operand_refs} '
        f'{attrs} : ({operand_types}) -> {_mlir_type(result.type)}'
    )
    return result


EMITTERS = EmitRegistry({}, functional_emitter)


@mlir_emitter(op=weight)
def weight_emitter(
    builder: MLIRModuleBuilder,
    op: OpDef,
    kwargs: Mapping[str, Any],
) -> SSAValue:
    sym_name = _require_str(kwargs, 'sym_name')
    storage_key = _require_str(kwargs, 'storage_key')
    layout = _require_str(kwargs, 'layout')
    packing = kwargs.get('packing', dense_packing())
    result_type = _require_type(kwargs, 'result_type', WeightType)
    result = builder._result_value(op, result_type, kwargs)
    builder._ops.append(
        f'{result.ref()} = {LATTICE_DIALECT.qualified_op_name(op)} '
        f'@{sym_name} '
        f'{{storage_key = "{storage_key}", '
        f'layout = #lattice.weight_layout<{layout}>, '
        f'packing = {_packing(packing)}}} : {_mlir_type(result.type)}'
    )
    return result


@mlir_emitter(op=sparse_decompose)
def sparse_decompose_emitter(
    builder: MLIRModuleBuilder,
    op: OpDef,
    kwargs: Mapping[str, Any],
) -> tuple[SSAValue, ...]:
    input_value = _require_value(kwargs, 'input')
    result_types = _require_sequence(kwargs, 'result_types')
    if len(result_types) != 3:
        raise ValueError('sparse_decompose requires three result types.')
    results = tuple(
        builder._result_value(op, _type(result_type), kwargs, index=index)
        for index, result_type in enumerate(result_types)
    )
    refs = ', '.join(result.ref() for result in results)
    result_type_text = ', '.join(
        _mlir_type(result.type) for result in results
    )
    builder._ops.append(
        f'{refs} = {LATTICE_DIALECT.qualified_op_name(op)} '
        f'{input_value.ref()} : '
        f'{_mlir_type(input_value.type)} -> ({result_type_text})'
    )
    return results


def _type(
    value: str | TensorType | SparseTensorType | WeightType,
) -> TensorType | SparseTensorType | WeightType:
    if isinstance(value, TensorType | SparseTensorType | WeightType):
        return value
    return TensorType(value)


def _mlir_type(value: TensorType | SparseTensorType | WeightType) -> str:
    return value.mlir()


def _default_output_role(value: SSAValue) -> OutputRole:
    if isinstance(value.type, SparseTensorType):
        return 'sparse_tensor'
    return 'tensor'


def _string_array(values: Iterable[str]) -> str:
    return (
        '['
        + ', '.join(f'"{_escape_string(value)}"' for value in values)
        + ']'
    )


def _escape_string(value: str) -> str:
    return value.replace('\\', '\\\\').replace('"', '\\"')


def _require_value(kwargs: Mapping[str, Any], name: str) -> SSAValue:
    value = _require_any(kwargs, name)
    if not isinstance(value, SSAValue):
        raise TypeError(f'{name} must be an SSAValue.')
    return value


def _require_str(kwargs: Mapping[str, Any], name: str) -> str:
    value = _require_any(kwargs, name)
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string.')
    return value


def _require_type[T](
    kwargs: Mapping[str, Any],
    name: str,
    expected: type[T],
) -> T:
    value = _require_any(kwargs, name)
    if not isinstance(value, expected):
        raise TypeError(f'{name} must be {expected.__name__}.')
    return value


def _require_any(kwargs: Mapping[str, Any], name: str) -> Any:
    try:
        return kwargs[name]
    except KeyError as exc:
        raise ValueError(
            f'missing required MLIR builder argument: {name}'
        ) from exc


def _require_sequence(
    kwargs: Mapping[str, Any], name: str
) -> Sequence[Any]:
    value = _require_any(kwargs, name)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f'{name} must be a sequence.')
    return value


def _format_attrs(op: OpDef, kwargs: Mapping[str, Any]) -> str:
    if not op.attributes:
        return ''
    attrs = []
    for attr in op.attributes:
        if attr.name not in kwargs:
            if attr.required:
                raise ValueError(
                    f'missing required MLIR builder argument: {attr.name}'
                )
            continue
        attrs.append(
            f'{attr.name} = {_format_attr(attr.kind, kwargs[attr.name])}'
        )
    return '{' + ', '.join(attrs) + '}'


def _format_attr(kind: str, value: Any) -> str:
    if kind == 'coord':
        return f'#lattice.coord<{value}>'
    if kind == 'weight_layout':
        return f'#lattice.weight_layout<{value}>'
    if kind == 'packing':
        return _packing(value)
    if kind == 'activation':
        return f'#lattice.activation<{value}>'
    if kind == 'gelu_approx':
        return f'#lattice.gelu_approx<{value}>'
    if kind == 'join':
        return f'#lattice.join<{value}>'
    if kind == 'binary_op':
        return f'#lattice.binary_op<{value}>'
    if kind == 'pool_mode':
        return f'#lattice.pool_mode<{value}>'
    if kind == 'voxel_reduction':
        return f'#lattice.voxel_reduction<{value}>'
    if kind == 'point_interpolation':
        return f'#lattice.point_interpolation<{value}>'
    if kind == 'i64_triple':
        return _triple(value)
    if kind == 'f64_triple':
        return _float_triple(value)
    if kind == 'i64':
        return str(int(value))
    if kind == 'f32':
        return f'{_decimal_float(value)} : f32'
    if kind == 'str':
        return f'"{value}"'
    raise ValueError(f'unsupported MLIR attribute kind: {kind}')


def _triple(value: Any) -> str:
    if isinstance(value, int):
        items = (value, value, value)
    else:
        items = tuple(int(item) for item in value)
    if len(items) != 3:
        raise ValueError(
            'MLIR triple attributes require exactly 3 integers.'
        )
    return f'array<i64: {items[0]}, {items[1]}, {items[2]}>'


def _float_triple(value: Any) -> str:
    if isinstance(value, int | float):
        items = (float(value), float(value), float(value))
    else:
        items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError(
            'MLIR float triple attributes require exactly 3 values.'
        )
    return f'array<f64: {items[0]}, {items[1]}, {items[2]}>'


def _decimal_float(value: Any) -> str:
    text = f'{float(value):.12f}'.rstrip('0').rstrip('.')
    if text in ('', '-0'):
        return '0.0'
    if '.' not in text:
        return f'{text}.0'
    return text


def _packing(value: Any) -> str:
    if not isinstance(value, Packing):
        raise TypeError('packing must be a Packing object.')
    return value.mlir()

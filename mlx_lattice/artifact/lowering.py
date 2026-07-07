from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    runtime_checkable,
)

import mlx.core as mx
from lattice_contract import LATTICE_DIALECT, OpDef

from mlx_lattice.artifact.plan import PlanOperation
from mlx_lattice.core import QuantizedWeight, SparseTensor
from mlx_lattice.core.types import Triple

type RuntimeValue = SparseTensor | mx.array | PackedWeightPayload
type SparseJoin = Literal['inner', 'left', 'right', 'outer']
type QuantizedPackingKind = Literal['int4', 'int8']
type LoweringResult = RuntimeValue | tuple[RuntimeValue, ...]
type LatticeOpHandle = str | OpDef | Callable[..., Any]
type AttrParser[T] = Callable[[Mapping[str, Any], str], T]
ArtifactFnT = TypeVar('ArtifactFnT', bound=Callable[..., LoweringResult])


@dataclass(frozen=True, slots=True)
class PackedWeightPayload:
    """Packed artifact weight awaiting consumer-specific resolution.

    ``lattice.weight`` carries storage identity and packing metadata, but not
    convolution geometry or logical input-channel count. The consuming op
    supplies that semantic context before MLX execution receives a concrete
    :class:`~mlx_lattice.core.QuantizedWeight`.
    """

    storage_key: str
    layout: str
    kind: QuantizedPackingKind
    group_size: int
    scale_dtype: str
    mode: str
    weight: mx.array
    scales: mx.array
    biases: mx.array

    @property
    def bits(self) -> int:
        """Packed integer width in bits."""

        return 4 if self.kind == 'int4' else 8

    def resolve_linear(self, *, in_channels: int) -> QuantizedWeight:
        """Resolve this payload as a feature-linear weight."""

        if self.layout != 'linear_o_i':
            raise ValueError(
                'linear artifact weight must use linear_o_i layout.'
            )
        return self._resolve(
            in_channels=in_channels,
            kernel_size=(1, 1, 1),
            layout='linear',
        )

    def resolve_conv3d(
        self,
        *,
        in_channels: int,
        kernel_size: Triple,
    ) -> QuantizedWeight:
        """Resolve this payload as a sparse-convolution weight."""

        if self.layout != 'conv3d_o_zyx_i':
            raise ValueError(
                'conv3d artifact weight must use conv3d_o_zyx_i layout.'
            )
        return self._resolve(
            in_channels=in_channels,
            kernel_size=kernel_size,
            layout='dense_5d',
        )

    def _resolve(
        self,
        *,
        in_channels: int,
        kernel_size: Triple,
        layout: Literal['linear', 'dense_5d'],
    ) -> QuantizedWeight:
        if self.mode != 'affine':
            raise ValueError(
                f'unsupported quantized artifact packing mode: {self.mode}'
            )
        _require_scale_dtype(self.scales, self.scale_dtype)
        kernel_rows = kernel_size[0] * kernel_size[1] * kernel_size[2]
        if int(self.weight.shape[0]) != kernel_rows:
            raise ValueError(
                'packed artifact weight kernel rows do not match consumer '
                'kernel_size.'
            )
        out_channels = (
            int(self.weight.shape[1])
            if kernel_rows == 1
            else int(self.weight.shape[2])
        )
        return QuantizedWeight(
            weight=self.weight,
            scales=self.scales,
            biases=self.biases,
            group_size=self.group_size,
            bits=self.bits,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            layout=layout,
        )


class LoweringProgram(Protocol):
    """Program context required by artifact lowerings."""

    weights: Mapping[str, mx.array]


class LoweringFn(Protocol):
    """Runtime lowering callable with a stable function name."""

    __name__: str

    def __call__(
        self,
        program: LoweringProgram,
        operation: PlanOperation,
        values: dict[str, RuntimeValue],
    ) -> LoweringResult: ...


@dataclass(frozen=True, slots=True)
class LoweringContext:
    """Concrete artifact execution context for one MLIR operation."""

    program: LoweringProgram
    operation: PlanOperation
    values: dict[str, RuntimeValue]


@runtime_checkable
class ArtifactParam(Protocol):
    """Annotation metadata that binds one Python parameter from MLIR."""

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
        name: str,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ProgramParam:
    """Bind the current artifact program."""

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
        name: str,
    ) -> LoweringProgram:
        del resolved, name
        return context.program


@dataclass(frozen=True, slots=True)
class OperandParam:
    """Bind an SSA operand from the runtime value table."""

    index: int
    reader: Callable[[Mapping[str, RuntimeValue], str], Any]
    optional: bool = False

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
        name: str,
    ) -> Any:
        del resolved, name
        if len(context.operation.operands) <= self.index:
            if self.optional:
                return None
            raise ValueError(
                f'{context.operation.name} missing operand {self.index}.'
            )
        return self.reader(
            context.values,
            context.operation.operands[self.index],
        )


@dataclass(frozen=True, slots=True)
class ConvWeightParam:
    """Bind and resolve a sparse-convolution weight operand."""

    index: int
    input: str
    kernel_size: str

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
        name: str,
    ) -> mx.array | QuantizedWeight:
        del name
        x = resolved[self.input]
        kernel_size = resolved.get(self.kernel_size)
        if kernel_size is None:
            kernel_size = triple_attr(
                context.operation.attrs,
                self.kernel_size,
            )
        if not isinstance(x, SparseTensor):
            raise TypeError(
                f'{context.operation.name} conv weight input '
                f'{self.input!r} must be a SparseTensor.'
            )
        return conv_weight(
            context.values,
            context.operation.operands[self.index],
            in_channels=x.channels,
            kernel_size=kernel_size,
        )


@dataclass(frozen=True, slots=True)
class LinearWeightParam:
    """Bind and resolve a dense-feature linear weight operand."""

    index: int
    input: str

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
        name: str,
    ) -> mx.array | QuantizedWeight:
        del name
        x = resolved[self.input]
        if isinstance(x, SparseTensor | PackedWeightPayload):
            raise TypeError(
                f'{context.operation.name} linear weight input '
                f'{self.input!r} must be an MLX array.'
            )
        return linear_weight(
            context.values,
            context.operation.operands[self.index],
            in_channels=int(x.shape[1]),
        )


@dataclass(frozen=True, slots=True)
class AttrParam:
    """Bind and parse a named MLIR operation attribute."""

    parser: AttrParser[Any]
    attr_name: str | None = None

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
        name: str,
    ) -> Any:
        del resolved
        return self.parser(context.operation.attrs, self.attr_name or name)


@dataclass(slots=True)
class LoweringRegistry:
    """Annotation-backed MLX lowering registry for lattice MLIR ops."""

    functions: dict[str, LoweringFn]

    def register(self, op: LatticeOpHandle, fn: LoweringFn) -> LoweringFn:
        """Register ``fn`` as the MLX artifact lowering for ``op``."""

        definition = LATTICE_DIALECT.resolve_op(op)
        name = LATTICE_DIALECT.qualified_op_name(definition)
        if name in self.functions:
            raise ValueError(f'duplicate artifact lowering: {name}')
        self.functions[name] = fn
        return fn

    def lower(
        self,
        program: LoweringProgram,
        operation: PlanOperation,
        values: dict[str, RuntimeValue],
    ) -> LoweringResult:
        """Run the registered lowering for one native MLIR plan operation."""

        try:
            lowering = self.functions[operation.name]
        except KeyError as exc:
            raise ValueError(
                f'unsupported artifact lowering op: {operation.name}'
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
) -> Callable[[ArtifactFnT], ArtifactFnT]:
    """Bind an artifact lowering to a lattice op or API declaration."""

    target = dialect_op or op

    def decorator(fn: ArtifactFnT) -> ArtifactFnT:
        definition = LATTICE_DIALECT.resolve_op(target)
        ARTIFACT_LOWERINGS.register(
            definition,
            _compile_artifact_lowering(fn, definition),
        )
        return fn

    return decorator


def program_param() -> ProgramParam:
    """Bind the current artifact program to an annotated parameter."""

    return ProgramParam()


def sparse_operand(index: int) -> OperandParam:
    """Bind a sparse tensor operand by SSA operand index."""

    return OperandParam(index, sparse)


def array_operand(index: int) -> OperandParam:
    """Bind an MLX array operand by SSA operand index."""

    return OperandParam(index, array)


def optional_array_operand(index: int) -> OperandParam:
    """Bind an optional MLX array operand by SSA operand index."""

    return OperandParam(index, array, optional=True)


def conv_weight_operand(
    index: int,
    *,
    input: str,
    kernel_size: str = 'kernel_size',
) -> ConvWeightParam:
    """Bind a dense or packed sparse-convolution weight operand."""

    return ConvWeightParam(index, input, kernel_size)


def linear_weight_operand(
    index: int,
    *,
    input: str,
) -> LinearWeightParam:
    """Bind a dense or packed feature-linear weight operand."""

    return LinearWeightParam(index, input)


def attr(parser: AttrParser[Any], name: str | None = None) -> AttrParam:
    """Bind an MLIR attribute parsed by ``parser``."""

    return AttrParam(parser, name)


def raw_attr(name: str | None = None) -> AttrParam:
    """Bind an MLIR attribute without conversion."""

    def read(attrs: Mapping[str, Any], attr_name: str) -> Any:
        return attrs[attr_name]

    return attr(read, name)


def str_attribute(name: str | None = None) -> AttrParam:
    """Bind a string MLIR attribute."""

    return attr(str_attr, name)


def float_attribute(name: str | None = None) -> AttrParam:
    """Bind a floating-point MLIR attribute."""

    return attr(float_attr, name)


def int_attribute(name: str | None = None) -> AttrParam:
    """Bind an integer MLIR attribute."""

    return attr(int_attr, name)


def triple_attribute(name: str | None = None) -> AttrParam:
    """Bind an integer triple MLIR attribute."""

    return attr(triple_attr, name)


def float_triple_attribute(name: str | None = None) -> AttrParam:
    """Bind a floating-point triple MLIR attribute."""

    return attr(float_triple_attr, name)


def join_attribute(name: str | None = None) -> AttrParam:
    """Bind a sparse join MLIR attribute."""

    return attr(join_attr, name)


def _compile_artifact_lowering(
    fn: Callable[..., LoweringResult],
    definition: OpDef,
) -> LoweringFn:
    fn_name = _callable_name(fn)
    parameters = signature(fn).parameters
    hints = get_type_hints(fn, include_extras=True)
    bindings: dict[str, ArtifactParam] = {}
    for name, parameter in parameters.items():
        if parameter.kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f'{fn_name} artifact lowering cannot use variadic '
                'parameters.'
            )
        binding = _artifact_binding(hints.get(name, parameter.annotation))
        if binding is None:
            raise TypeError(
                f'{fn_name}.{name} must be annotated with an artifact '
                'binding, for example Annotated[..., sparse_operand(0)].'
            )
        bindings[name] = binding
    _validate_artifact_bindings(fn_name, definition, bindings)

    def lower(
        program: LoweringProgram,
        operation: PlanOperation,
        values: dict[str, RuntimeValue],
    ) -> LoweringResult:
        context = LoweringContext(program, operation, values)
        resolved: dict[str, Any] = {}
        positional: list[Any] = []
        keywords: dict[str, Any] = {}
        for name, parameter in parameters.items():
            value = bindings[name].resolve(context, resolved, name)
            resolved[name] = value
            if parameter.kind is Parameter.KEYWORD_ONLY:
                keywords[name] = value
            else:
                positional.append(value)
        return fn(*positional, **keywords)

    lower.__name__ = fn_name
    lower.__dict__['__artifact_source__'] = fn
    lower.__dict__['__artifact_bindings__'] = dict(bindings)
    return cast(LoweringFn, lower)


def _validate_artifact_bindings(
    fn_name: str,
    definition: OpDef,
    bindings: Mapping[str, ArtifactParam],
) -> None:
    ssa_operands = tuple(
        operand for operand in definition.operands if operand.kind == 'ssa'
    )
    allowed_attrs = {attr.name for attr in definition.attributes} | {
        operand.name
        for operand in definition.operands
        if operand.kind == 'symbol'
    }
    for name, binding in bindings.items():
        if isinstance(binding, OperandParam):
            _validate_operand_index(
                fn_name,
                definition,
                name,
                binding.index,
                len(ssa_operands),
            )
        elif isinstance(binding, ConvWeightParam):
            _validate_operand_index(
                fn_name,
                definition,
                name,
                binding.index,
                len(ssa_operands),
            )
            _validate_dependency(fn_name, name, binding.input, bindings)
            _validate_attr_name(
                fn_name,
                definition,
                name,
                binding.kernel_size,
                allowed_attrs,
            )
        elif isinstance(binding, LinearWeightParam):
            _validate_operand_index(
                fn_name,
                definition,
                name,
                binding.index,
                len(ssa_operands),
            )
            _validate_dependency(fn_name, name, binding.input, bindings)
        elif isinstance(binding, AttrParam):
            _validate_attr_name(
                fn_name,
                definition,
                name,
                binding.attr_name or name,
                allowed_attrs,
            )


def _validate_operand_index(
    fn_name: str,
    definition: OpDef,
    param_name: str,
    index: int,
    operand_count: int,
) -> None:
    if not 0 <= index < operand_count:
        raise ValueError(
            f'{fn_name}.{param_name} binds operand {index}, but '
            f'lattice.{definition.name} has {operand_count} SSA operands.'
        )


def _validate_dependency(
    fn_name: str,
    param_name: str,
    dependency: str,
    bindings: Mapping[str, ArtifactParam],
) -> None:
    if dependency not in bindings:
        raise ValueError(
            f'{fn_name}.{param_name} depends on missing artifact parameter '
            f'{dependency!r}.'
        )


def _validate_attr_name(
    fn_name: str,
    definition: OpDef,
    param_name: str,
    attr_name: str,
    allowed_attrs: set[str],
) -> None:
    if attr_name not in allowed_attrs:
        raise ValueError(
            f'{fn_name}.{param_name} binds attr {attr_name!r}, but '
            f'lattice.{definition.name} does not declare it.'
        )


def _artifact_binding(annotation: Any) -> ArtifactParam | None:
    if get_origin(annotation) is not Annotated:
        return None
    for metadata in get_args(annotation)[1:]:
        if isinstance(metadata, ArtifactParam):
            return metadata
    return None


def _callable_name(fn: Callable[..., Any]) -> str:
    name = cast(Any, fn).__name__
    if not isinstance(name, str):
        raise TypeError('lattice lowering declarations must be named.')
    return name


def results(operation: PlanOperation) -> tuple[str, ...]:
    """Return operation result value names."""

    return operation.results


def sparse(values: Mapping[str, RuntimeValue], name: str) -> SparseTensor:
    """Read a sparse value from the runtime environment."""

    value = values[name]
    if not isinstance(value, SparseTensor):
        raise TypeError(f'{name} must be a SparseTensor.')
    return value


def array(values: Mapping[str, RuntimeValue], name: str) -> mx.array:
    """Read an MLX array from the runtime environment."""

    value = values[name]
    if isinstance(value, SparseTensor | PackedWeightPayload):
        raise TypeError(f'{name} must be an MLX array.')
    return value


def conv_weight(
    values: Mapping[str, RuntimeValue],
    name: str,
    *,
    in_channels: int,
    kernel_size: Triple,
) -> mx.array | QuantizedWeight:
    """Read and resolve a dense or packed sparse-convolution weight."""

    value = values[name]
    if isinstance(value, mx.array):
        return value
    if isinstance(value, PackedWeightPayload):
        return value.resolve_conv3d(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
    raise TypeError(f'{name} must be a weight value.')


def linear_weight(
    values: Mapping[str, RuntimeValue],
    name: str,
    *,
    in_channels: int,
) -> mx.array | QuantizedWeight:
    """Read and resolve a dense or packed feature-linear weight."""

    value = values[name]
    if isinstance(value, mx.array):
        return value
    if isinstance(value, PackedWeightPayload):
        return value.resolve_linear(in_channels=in_channels)
    raise TypeError(f'{name} must be a weight value.')


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


def int_attr(attrs: Mapping[str, Any], name: str) -> int:
    """Read an integer attribute."""

    value = attrs[name]
    if not isinstance(value, int):
        raise TypeError(f'{name} must be an integer attribute.')
    return value


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


def float_triple_attr(
    attrs: Mapping[str, Any], name: str
) -> tuple[float, float, float]:
    """Read a floating-point triple attribute."""

    value = attrs[name]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f'{name} must be a floating-point triple.')
    items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError(f'{name} must contain exactly 3 values.')
    return cast(tuple[float, float, float], items)


def join_attr(attrs: Mapping[str, Any], name: str) -> SparseJoin:
    """Read a sparse join attribute."""

    value = str_attr(attrs, name)
    if value not in ('inner', 'left', 'right', 'outer'):
        raise ValueError(f'unsupported sparse join mode: {value}')
    return cast(SparseJoin, value)


def packing_kind(value: Any) -> QuantizedPackingKind:
    """Validate a quantized packing kind."""

    if value not in ('int4', 'int8'):
        raise ValueError(f'unsupported quantized packing kind: {value}')
    return cast(QuantizedPackingKind, value)


def _require_scale_dtype(array: mx.array, expected: str) -> None:
    actual = _dtype_name(array.dtype)
    if actual != expected:
        raise ValueError(
            f'quantized artifact scales dtype mismatch: expected '
            f'{expected}, found {actual}.'
        )


def _dtype_name(dtype: mx.Dtype) -> str:
    if dtype == mx.float16:
        return 'f16'
    if dtype == mx.float32:
        return 'f32'
    return str(dtype)

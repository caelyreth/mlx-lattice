from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Literal, Protocol, TypeVar, cast, overload

import mlx.core as mx
from lattice_contract import LATTICE_DIALECT, OpDef
from lattice_contract.schema import OpAttributeDef, OperandDef

from mlx_lattice.artifact.plan import PlanOperation
from mlx_lattice.core import QuantizedWeight, SparseTensor
from mlx_lattice.core.types import Triple

type RuntimeValue = SparseTensor | mx.array | PackedWeightPayload
type SparseJoin = Literal['inner', 'left', 'right', 'outer']
type QuantizedPackingKind = Literal['int4', 'int8']
type LoweringResult = RuntimeValue | tuple[RuntimeValue, ...]
type LatticeOpHandle = str | OpDef | Callable[..., Any]
ArtifactFnT = TypeVar('ArtifactFnT', bound=Callable[..., LoweringResult])


@dataclass(frozen=True, slots=True)
class PackedWeightPayload:
    """Packed artifact weight awaiting consumer-specific resolution."""

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

        if self.layout != 'conv3d_o_xyz_i':
            raise ValueError(
                'conv3d artifact weight must use conv3d_o_xyz_i layout.'
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

    @property
    def weights(self) -> Mapping[str, mx.array]: ...


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


@dataclass(frozen=True, slots=True)
class WeightResolver:
    """Resolve a symbolic artifact weight for a consumer operation."""

    kind: Literal['conv3d', 'linear']
    input: str
    kernel_size: str = 'kernel_size'

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
        operand_name: str,
        value: RuntimeValue,
    ) -> mx.array | QuantizedWeight:
        if isinstance(value, mx.array):
            return value
        if not isinstance(value, PackedWeightPayload):
            raise TypeError(f'{operand_name} must be a weight value.')
        input_value = resolved[self.input]
        if self.kind == 'conv3d':
            if not isinstance(input_value, SparseTensor):
                raise TypeError(
                    f'{context.operation.name} conv weight input '
                    f'{self.input!r} must be a SparseTensor.'
                )
            kernel_size = resolved.get(self.kernel_size)
            if kernel_size is None:
                kernel_size = _read_attr(
                    context.operation.attrs,
                    self.kernel_size,
                    'i64_triple',
                )
            return value.resolve_conv3d(
                in_channels=input_value.channels,
                kernel_size=kernel_size,
            )
        if isinstance(input_value, SparseTensor | PackedWeightPayload):
            raise TypeError(
                f'{context.operation.name} linear weight input '
                f'{self.input!r} must be an MLX array.'
            )
        return value.resolve_linear(in_channels=int(input_value.shape[1]))


@dataclass(frozen=True, slots=True)
class _Binding:
    name: str

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
    ) -> Any:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _ProgramBinding(_Binding):
    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
    ) -> LoweringProgram:
        del resolved
        return context.program


@dataclass(frozen=True, slots=True)
class _OperandBinding(_Binding):
    operand: OperandDef
    index: int
    resolver: WeightResolver | None = None

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
    ) -> Any:
        if len(context.operation.operands) <= self.index:
            if self.operand.optional:
                return None
            raise ValueError(
                f'{context.operation.name} missing operand {self.index}.'
            )
        value = context.values[context.operation.operands[self.index]]
        if self.resolver is not None:
            return self.resolver.resolve(
                context,
                resolved,
                self.operand.name,
                value,
            )
        if self.operand.type == 'sparse_tensor':
            if not isinstance(value, SparseTensor):
                raise TypeError(
                    f'{self.operand.name} must be a SparseTensor.'
                )
        elif self.operand.type == 'tensor' and isinstance(
            value, SparseTensor | PackedWeightPayload
        ):
            raise TypeError(f'{self.operand.name} must be an MLX array.')
        return value


@dataclass(frozen=True, slots=True)
class _AttrBinding(_Binding):
    attr: OpAttributeDef

    def resolve(
        self,
        context: LoweringContext,
        resolved: Mapping[str, Any],
    ) -> Any:
        del resolved
        return _read_attr(
            context.operation.attrs,
            self.attr.name,
            self.attr.kind,
        )


@dataclass(slots=True)
class LoweringRegistry:
    """Schema-backed MLX lowering registry for lattice MLIR ops."""

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
    weights: Mapping[str, WeightResolver] | None = None,
) -> Callable[[ArtifactFnT], ArtifactFnT]:
    """Bind an artifact lowering to a lattice op or API declaration.

    Operands and attributes are inferred from the lattice dialect schema by
    matching Python parameter names to MLIR operand/attribute names. Explicit
    ``weights`` metadata is only needed when a symbolic weight may be packed and
    requires consumer context before execution.
    """

    target = dialect_op or op

    def decorator(fn: ArtifactFnT) -> ArtifactFnT:
        definition = LATTICE_DIALECT.resolve_op(target)
        ARTIFACT_LOWERINGS.register(
            definition,
            _compile_artifact_lowering(fn, definition, weights=weights),
        )
        return fn

    return decorator


def conv_weight(
    *,
    input: str = 'input',
    kernel_size: str = 'kernel_size',
) -> WeightResolver:
    """Declare a packed-capable sparse-convolution weight parameter."""

    return WeightResolver('conv3d', input=input, kernel_size=kernel_size)


def linear_weight(*, input: str = 'input') -> WeightResolver:
    """Declare a packed-capable dense-feature linear weight parameter."""

    return WeightResolver('linear', input=input)


def _compile_artifact_lowering(
    fn: Callable[..., LoweringResult],
    definition: OpDef,
    *,
    weights: Mapping[str, WeightResolver] | None = None,
) -> LoweringFn:
    fn_name = _callable_name(fn)
    parameters = signature(fn).parameters
    weight_resolvers = dict(weights or {})
    bindings = _compile_bindings(
        fn_name,
        definition,
        parameters,
        weight_resolvers,
    )

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
            value = bindings[name].resolve(context, resolved)
            resolved[name] = value
            if parameter.kind is Parameter.KEYWORD_ONLY:
                keywords[name] = value
            else:
                positional.append(value)
        return fn(*positional, **keywords)

    lower.__name__ = fn_name
    return cast(LoweringFn, lower)


def _compile_bindings(
    fn_name: str,
    definition: OpDef,
    parameters: Mapping[str, Parameter],
    weight_resolvers: Mapping[str, WeightResolver],
) -> dict[str, _Binding]:
    ssa_operands = tuple(
        operand for operand in definition.operands if operand.kind == 'ssa'
    )
    operands_by_name = {operand.name: operand for operand in ssa_operands}
    operand_indices = {
        operand.name: index for index, operand in enumerate(ssa_operands)
    }
    attrs_by_name = {attr.name: attr for attr in definition.attributes}
    attrs_by_name.update(
        {
            operand.name: OpAttributeDef(operand.name, 'str')
            for operand in definition.operands
            if operand.kind == 'symbol'
        }
    )

    bindings: dict[str, _Binding] = {}
    for name, parameter in parameters.items():
        if parameter.kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f'{fn_name} artifact lowering cannot use variadic '
                'parameters.'
            )
        if name == 'program':
            bindings[name] = _ProgramBinding(name)
            continue
        if name in operands_by_name:
            bindings[name] = _OperandBinding(
                name,
                operand=operands_by_name[name],
                index=operand_indices[name],
                resolver=weight_resolvers.get(name),
            )
            continue
        if name in attrs_by_name:
            bindings[name] = _AttrBinding(name, attrs_by_name[name])
            continue
        raise ValueError(
            f'{fn_name}.{name} is not declared by '
            f'lattice.{definition.name}.'
        )

    _validate_weight_resolvers(
        fn_name,
        definition,
        bindings,
        operands_by_name,
        weight_resolvers,
    )
    return bindings


def _validate_weight_resolvers(
    fn_name: str,
    definition: OpDef,
    bindings: Mapping[str, _Binding],
    operands_by_name: Mapping[str, OperandDef],
    weight_resolvers: Mapping[str, WeightResolver],
) -> None:
    for name, resolver in weight_resolvers.items():
        operand = operands_by_name.get(name)
        if operand is None:
            raise ValueError(
                f'{fn_name}.{name} weight resolver targets an unknown '
                f'lattice.{definition.name} operand.'
            )
        if operand.type != 'weight':
            raise ValueError(
                f'{fn_name}.{name} weight resolver can only target weight '
                'operands.'
            )
        if name not in bindings:
            raise ValueError(
                f'{fn_name}.{name} weight resolver has no matching function '
                'parameter.'
            )
        if resolver.input not in bindings:
            raise ValueError(
                f'{fn_name}.{name} depends on missing artifact parameter '
                f'{resolver.input!r}.'
            )
        if (
            resolver.kind == 'conv3d'
            and resolver.kernel_size not in bindings
            and resolver.kernel_size
            not in {attr.name for attr in definition.attributes}
        ):
            raise ValueError(
                f'{fn_name}.{name} depends on missing kernel_size '
                f'{resolver.kernel_size!r}.'
            )


def _callable_name(fn: Callable[..., Any]) -> str:
    name = cast(Any, fn).__name__
    if not isinstance(name, str):
        raise TypeError('lattice lowering declarations must be named.')
    return name


def results(operation: PlanOperation) -> tuple[str, ...]:
    """Return operation result value names."""

    return operation.results


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
    return items[0], items[1], items[2]


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
    return items[0], items[1], items[2]


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


_ATTR_READERS: Mapping[str, Callable[[Mapping[str, Any], str], Any]] = {
    'coord': str_attr,
    'feature_layout': str_attr,
    'weight_layout': str_attr,
    'packing': lambda attrs, name: attrs[name],
    'activation': str_attr,
    'gelu_approx': str_attr,
    'join': join_attr,
    'binary_op': str_attr,
    'pool_mode': str_attr,
    'voxel_reduction': str_attr,
    'point_interpolation': str_attr,
    'i64_triple': triple_attr,
    'f64_triple': float_triple_attr,
    'i64': int_attr,
    'f32': float_attr,
    'str': str_attr,
}


def _read_attr(attrs: Mapping[str, Any], name: str, kind: str) -> Any:
    try:
        reader = _ATTR_READERS[kind]
    except KeyError as exc:
        raise ValueError(
            f'unsupported MLIR attribute kind: {kind}'
        ) from exc
    return reader(attrs, name)


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

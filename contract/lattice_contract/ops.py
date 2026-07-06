from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from lattice_contract.manifest import (
    IRInputRef,
    IRNode,
    IRValueType,
    ir_value_type,
)

DeclarationT = TypeVar('DeclarationT', bound=Callable)


class IRParameterKind(StrEnum):
    """Storage kind for a persisted operation parameter."""

    ARRAY = 'array'
    OPTIONAL_ARRAY = 'optional_array'
    QUANTIZED_WEIGHT = 'quantized_weight'
    ARRAY_OR_QUANTIZED_WEIGHT = 'array_or_quantized_weight'


@dataclass(frozen=True, slots=True)
class IROpSpec:
    """Static graph schema for one lattice IR operation."""

    name: str
    inputs: frozenset[str]
    outputs: frozenset[str]
    output_types: dict[str, IRValueType]
    input_types: dict[str, IRValueType] = field(default_factory=dict)
    value_attribute_types: dict[str, IRValueType] = field(
        default_factory=dict
    )
    parameters: frozenset[str] = frozenset()
    optional_parameters: frozenset[str] = frozenset()
    attributes: frozenset[str] = frozenset()
    value_attributes: frozenset[str] = frozenset()
    requires_support: bool = False


@dataclass(frozen=True, slots=True)
class IROpContract:
    """Canonical semantic contract for one lattice IR operation."""

    spec: IROpSpec
    parameter_kinds: Mapping[str, IRParameterKind] = field(
        default_factory=dict
    )
    optional_parameter_kinds: Mapping[str, IRParameterKind] = field(
        default_factory=dict
    )

    @property
    def name(self) -> str:
        """Return the manifest operation name."""

        return self.spec.name


_OP_CONTRACTS: dict[str, IROpContract] = {}


def ir_op_contract(
    name: str,
    *,
    inputs: set[str] | frozenset[str],
    outputs: set[str] | frozenset[str],
    output_types: Mapping[str, str] | None = None,
    input_types: Mapping[str, str] | None = None,
    value_attribute_types: Mapping[str, str] | None = None,
    parameters: set[str] | frozenset[str] | None = None,
    optional_parameters: set[str] | frozenset[str] | None = None,
    attributes: set[str] | frozenset[str] | None = None,
    value_attributes: set[str] | frozenset[str] | None = None,
    parameter_kinds: Mapping[str, str | IRParameterKind] | None = None,
    optional_parameter_kinds: Mapping[str, str | IRParameterKind]
    | None = None,
    requires_support: bool = False,
) -> IROpContract:
    """Create an unregistered lattice IR operation contract."""

    required_params = frozenset(parameters or ())
    optional_params = frozenset(optional_parameters or ())
    return IROpContract(
        spec=IROpSpec(
            name=name,
            inputs=frozenset(inputs),
            outputs=frozenset(outputs),
            output_types=_value_type_map(output_types),
            input_types=_value_type_map(input_types),
            value_attribute_types=_value_type_map(value_attribute_types),
            parameters=required_params,
            optional_parameters=optional_params,
            attributes=frozenset(attributes or ()),
            value_attributes=frozenset(value_attributes or ()),
            requires_support=requires_support,
        ),
        parameter_kinds=_checked_parameter_kinds(
            parameter_kinds, required_params, f'{name}.parameter_kinds'
        ),
        optional_parameter_kinds=_checked_parameter_kinds(
            optional_parameter_kinds,
            optional_params,
            f'{name}.optional_parameter_kinds',
        ),
    )


def register_op_contract(contract: IROpContract) -> IROpContract:
    """Register an extension operation contract and return it."""

    _register(contract)
    return contract


def ir_op_spec(
    name: str,
    *,
    inputs: set[str],
    outputs: set[str],
    output_types: Mapping[str, str] | None = None,
    input_types: Mapping[str, str] | None = None,
    value_attribute_types: Mapping[str, str] | None = None,
    parameters: set[str] | None = None,
    optional_parameters: set[str] | None = None,
    attributes: set[str] | None = None,
    value_attributes: set[str] | None = None,
    requires_support: bool = False,
) -> Callable[[DeclarationT], DeclarationT]:
    """Register an extension operation with a compact annotation."""

    contract = ir_op_contract(
        name,
        inputs=inputs,
        outputs=outputs,
        output_types=output_types,
        input_types=input_types,
        value_attribute_types=value_attribute_types,
        parameters=parameters,
        optional_parameters=optional_parameters,
        attributes=attributes,
        value_attributes=value_attributes,
        requires_support=requires_support,
    )

    def decorator(declaration: DeclarationT) -> DeclarationT:
        _register(contract)
        return declaration

    return decorator


def iter_op_specs() -> Iterator[IROpSpec]:
    """Iterate registered IR operation specs."""

    return (contract.spec for contract in _OP_CONTRACTS.values())


def iter_op_contracts() -> Iterator[IROpContract]:
    """Iterate registered canonical IR operation contracts."""

    return iter(_OP_CONTRACTS.values())


def op_spec(name: str) -> IROpSpec:
    """Return the registered spec for ``name`` or raise ``ValueError``."""

    return op_contract(name).spec


def op_contract(name: str) -> IROpContract:
    """Return the registered operation contract for ``name``."""

    try:
        return _OP_CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f'unsupported lattice IR op: {name!r}.') from exc


def validate_node_against_spec(node: IRNode) -> None:
    """Validate node ports, parameters, and attributes against its spec."""

    spec = op_spec(node.op)
    _require_keys(node.inputs, spec.inputs, f'{node.id}.inputs')
    _require_keys(node.outputs, spec.outputs, f'{node.id}.outputs')
    _require_subset(
        spec.parameters,
        set(node.parameters),
        f'{node.id}.parameters',
        required=True,
    )
    _require_subset(
        set(node.parameters),
        spec.parameters | spec.optional_parameters,
        f'{node.id}.parameters',
    )
    _require_subset(
        set(node.attributes),
        spec.attributes | spec.value_attributes,
        f'{node.id}.attributes',
    )
    if spec.requires_support and node.support is None:
        raise ValueError(f'{node.id} requires a support object.')


def _parameter_kinds(
    values: Mapping[str, str | IRParameterKind] | None,
) -> dict[str, IRParameterKind]:
    return {
        name: IRParameterKind(value)
        for name, value in dict(values or {}).items()
    }


def _checked_parameter_kinds(
    values: Mapping[str, str | IRParameterKind] | None,
    allowed: frozenset[str],
    path: str,
) -> dict[str, IRParameterKind]:
    out = _parameter_kinds(values)
    _require_subset(set(out), allowed, path)
    return out


def _value_type_map(
    values: Mapping[str, str] | None,
) -> dict[str, IRValueType]:
    return {
        name: ir_value_type(value)
        for name, value in dict(values or {}).items()
    }


def _register(contract: IROpContract) -> None:
    if contract.name in _OP_CONTRACTS:
        raise ValueError(
            f'duplicate lattice IR op registration: {contract.name}.'
        )
    _OP_CONTRACTS[contract.name] = contract


def _require_keys(
    values: Mapping[str, IRInputRef],
    expected: frozenset[str],
    path: str,
) -> None:
    actual = set(values)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(
            f'{path} missing required keys: {sorted(missing)}.'
        )
    if extra:
        raise ValueError(f'{path} has unsupported keys: {sorted(extra)}.')


def _require_subset(
    actual: set[str] | frozenset[str],
    allowed: set[str] | frozenset[str],
    path: str,
    *,
    required: bool = False,
) -> None:
    delta = allowed - actual if required else actual - allowed
    if not delta:
        return
    label = 'missing required' if required else 'has unsupported'
    raise ValueError(f'{path} {label} keys: {sorted(delta)}.')


def _builtin(
    name: str,
    *,
    inputs: set[str],
    outputs: set[str],
    output_types: Mapping[str, str] | None = None,
    input_types: Mapping[str, str] | None = None,
    value_attribute_types: Mapping[str, str] | None = None,
    parameters: set[str] | None = None,
    optional_parameters: set[str] | None = None,
    attributes: set[str] | None = None,
    value_attributes: set[str] | None = None,
    parameter_kinds: Mapping[str, str | IRParameterKind] | None = None,
    optional_parameter_kinds: Mapping[str, str | IRParameterKind]
    | None = None,
) -> IROpContract:
    return register_op_contract(
        ir_op_contract(
            name,
            inputs=inputs,
            outputs=outputs,
            output_types=output_types,
            input_types=input_types,
            value_attribute_types=value_attribute_types,
            parameters=parameters,
            optional_parameters=optional_parameters,
            attributes=attributes,
            value_attributes=value_attributes,
            parameter_kinds=parameter_kinds,
            optional_parameter_kinds=optional_parameter_kinds,
        )
    )


def _feature_unary(
    name: str,
    *,
    optional_parameters: set[str] | None = None,
    attributes: set[str] | None = None,
) -> IROpContract:
    return _builtin(
        name,
        inputs={'input'},
        outputs={'output'},
        input_types={'input': 'sparse_tensor'},
        output_types={'output': 'sparse_tensor'},
        optional_parameters=optional_parameters,
        attributes=attributes,
    )


def _global_pool(name: str) -> IROpContract:
    return _builtin(
        name,
        inputs={'input'},
        outputs={'output'},
        input_types={'input': 'sparse_tensor'},
        output_types={'output': 'dense_tensor'},
    )


def _local_pool(name: str, *, attributes: set[str]) -> IROpContract:
    return _builtin(
        name,
        inputs={'input'},
        outputs={'output'},
        input_types={'input': 'sparse_tensor'},
        output_types={'output': 'sparse_tensor'},
        attributes=attributes,
    )


# MARK: - built-in semantic contracts

VALUE_FIELD = _builtin(
    'value.field',
    inputs={'input'},
    outputs={'output'},
    output_types={'output': 'any'},
    attributes={'field'},
)
SPARSE_CONV3D = _builtin(
    'sparse.conv3d',
    inputs={'input'},
    outputs={'output'},
    input_types={'input': 'sparse_tensor'},
    output_types={'output': 'sparse_tensor'},
    parameters={'weight'},
    optional_parameters={'bias'},
    attributes={'kernel_size', 'stride', 'padding', 'dilation'},
    value_attributes={'coordinates'},
    parameter_kinds={'weight': IRParameterKind.ARRAY_OR_QUANTIZED_WEIGHT},
)
SPARSE_SUBM_CONV3D = _builtin(
    'sparse.subm_conv3d',
    inputs={'input'},
    outputs={'output'},
    input_types={'input': 'sparse_tensor'},
    output_types={'output': 'sparse_tensor'},
    parameters={'weight'},
    optional_parameters={'bias'},
    attributes={'kernel_size', 'dilation'},
    parameter_kinds={'weight': IRParameterKind.ARRAY_OR_QUANTIZED_WEIGHT},
)
SPARSE_CONV_TRANSPOSE3D = _builtin(
    'sparse.conv_transpose3d',
    inputs={'input'},
    outputs={'output'},
    input_types={'input': 'sparse_tensor'},
    output_types={'output': 'sparse_tensor'},
    parameters={'weight'},
    optional_parameters={'bias'},
    attributes={'kernel_size', 'stride', 'padding', 'dilation'},
    parameter_kinds={'weight': IRParameterKind.ARRAY_OR_QUANTIZED_WEIGHT},
)
SPARSE_GENERATIVE_CONV_TRANSPOSE3D = _builtin(
    'sparse.generative_conv_transpose3d',
    inputs={'input'},
    outputs={'output'},
    input_types={'input': 'sparse_tensor'},
    output_types={'output': 'sparse_tensor'},
    parameters={'weight'},
    optional_parameters={'bias'},
    attributes={'kernel_size', 'stride'},
    parameter_kinds={'weight': IRParameterKind.ARRAY_OR_QUANTIZED_WEIGHT},
)
SPARSE_ADD = _builtin(
    'sparse.add',
    inputs={'lhs', 'rhs'},
    outputs={'output'},
    input_types={'lhs': 'sparse_tensor', 'rhs': 'sparse_tensor'},
    output_types={'output': 'sparse_tensor'},
    attributes={'join'},
)
FEATURE_LINEAR = _builtin(
    'feature.linear',
    inputs={'input'},
    outputs={'output'},
    input_types={'input': 'any'},
    output_types={'output': 'any'},
    parameters={'weight'},
    optional_parameters={'bias'},
    parameter_kinds={'weight': IRParameterKind.ARRAY_OR_QUANTIZED_WEIGHT},
)
FEATURE_RELU = _feature_unary('feature.relu')
FEATURE_SIGMOID = _feature_unary('feature.sigmoid')
FEATURE_SILU = _feature_unary('feature.silu')
FEATURE_TANH = _feature_unary('feature.tanh')
FEATURE_GELU = _feature_unary('feature.gelu', attributes={'approximate'})
FEATURE_LEAKY_RELU = _feature_unary(
    'feature.leaky_relu',
    attributes={'negative_slope'},
)
FEATURE_SOFTPLUS = _feature_unary(
    'feature.softplus',
    attributes={'beta', 'threshold'},
)
FEATURE_DROPOUT = _feature_unary(
    'feature.dropout',
    attributes={'p', 'training'},
)
FEATURE_BATCH_NORM = _feature_unary(
    'feature.batch_norm',
    optional_parameters={'weight', 'bias', 'mean', 'var'},
    attributes={'eps'},
)
FEATURE_LAYER_NORM = _feature_unary(
    'feature.layer_norm',
    optional_parameters={'weight', 'bias'},
    attributes={'eps'},
)
FEATURE_RMS_NORM = _feature_unary(
    'feature.rms_norm',
    optional_parameters={'weight'},
    attributes={'eps'},
)
POOL3D = _local_pool(
    'pool.pool3d',
    attributes={'mode', 'kernel_size', 'stride', 'padding', 'dilation'},
)
POOL_SUM3D = _local_pool(
    'pool.sum3d',
    attributes={'kernel_size', 'stride', 'padding', 'dilation'},
)
POOL_MAX3D = _local_pool(
    'pool.max3d',
    attributes={'kernel_size', 'stride', 'padding', 'dilation'},
)
POOL_AVG3D = _local_pool(
    'pool.avg3d',
    attributes={'kernel_size', 'stride', 'padding', 'dilation'},
)
POOL_GLOBAL_SUM = _global_pool('pool.global_sum')
POOL_GLOBAL_AVG = _global_pool('pool.global_avg')
POOL_GLOBAL_MAX = _global_pool('pool.global_max')

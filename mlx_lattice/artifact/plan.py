from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from lattice_contract import (
    ARTIFACT_WEIGHT_FILE,
    CURRENT_DIALECT_VERSION,
    DIALECT_SCHEMA_DIGEST,
)
from lattice_contract.dialect import LATTICE_DIALECT
from lattice_contract.schema import OpDef


@dataclass(frozen=True, slots=True)
class PlanArgument:
    """One verified artifact entry argument.

    ``name`` is an importer-generated runtime value label. ``type`` is the
    canonical MLIR type text. ``role`` is MLIR-derived ABI metadata used for
    ergonomic bindings such as SparseTensor shorthand.
    """

    name: str
    abi_name: str
    type: str
    role: str


@dataclass(frozen=True, slots=True)
class PlanOutput:
    """One verified artifact entry output ABI item."""

    name: str
    abi_name: str
    type: str
    role: str


@dataclass(frozen=True, slots=True)
class PlanOperation:
    """One executable lattice operation in native importer order."""

    name: str
    definition: OpDef
    operands: tuple[str, ...]
    results: tuple[str, ...]
    attrs: Mapping[str, Any]
    operand_types: tuple[str, ...]
    result_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """Typed Python view of a native-verified lattice MLIR execution plan."""

    ir_version: int
    schema_digest: str
    weight_file: str
    name: str
    args: tuple[PlanArgument, ...]
    ops: tuple[PlanOperation, ...]
    returns: tuple[str, ...]
    outputs: tuple[PlanOutput, ...]

    @classmethod
    def from_native(cls, raw: Mapping[str, Any]) -> RuntimePlan:
        """Validate and freeze a native ``lattice_mlir_plan`` payload."""

        ir_version = _int(raw, 'ir_version')
        schema_digest = _str(raw, 'schema_digest')
        weight_file = _str(raw, 'weight_file')
        name = _str(raw, 'name')
        args = tuple(_argument(item) for item in _sequence(raw, 'args'))
        ops = tuple(_operation(item) for item in _sequence(raw, 'ops'))
        returns = tuple(str(item) for item in _sequence(raw, 'returns'))
        outputs = tuple(_output(item) for item in _sequence(raw, 'outputs'))
        _validate_plan_dataflow(
            ir_version=ir_version,
            schema_digest=schema_digest,
            weight_file=weight_file,
            name=name,
            args=args,
            ops=ops,
            returns=returns,
            outputs=outputs,
        )
        return cls(
            ir_version=ir_version,
            schema_digest=schema_digest,
            weight_file=weight_file,
            name=name,
            args=args,
            ops=ops,
            returns=returns,
            outputs=outputs,
        )


def _argument(raw: Any) -> PlanArgument:
    mapping = _mapping(raw, 'argument')
    return PlanArgument(
        name=_str(mapping, 'name'),
        abi_name=_str(mapping, 'abi_name'),
        type=_str(mapping, 'type'),
        role=_str(mapping, 'role'),
    )


def _output(raw: Any) -> PlanOutput:
    mapping = _mapping(raw, 'output')
    return PlanOutput(
        name=_str(mapping, 'name'),
        abi_name=_str(mapping, 'abi_name'),
        type=_str(mapping, 'type'),
        role=_str(mapping, 'role'),
    )


def _operation(raw: Any) -> PlanOperation:
    mapping = _mapping(raw, 'operation')
    name = _str(mapping, 'name')
    definition = LATTICE_DIALECT.resolve_qualified_op(name)
    operands = tuple(str(item) for item in _sequence(mapping, 'operands'))
    results = tuple(str(item) for item in _sequence(mapping, 'results'))
    attrs = _mapping(mapping.get('attrs', {}), 'attrs')
    operand_types = _optional_str_tuple(mapping, 'operand_types')
    result_types = _optional_str_tuple(mapping, 'result_types')
    if operand_types and len(operand_types) != len(operands):
        raise ValueError(
            f'{_str(mapping, "name")} operand_types must match operands.'
        )
    if result_types and len(result_types) != len(results):
        raise ValueError(
            f'{_str(mapping, "name")} result_types must match results.'
        )
    _validate_operation_shape(
        definition,
        operands=operands,
        results=results,
        attrs=attrs,
        name=name,
    )
    return PlanOperation(
        name=name,
        definition=definition,
        operands=operands,
        results=results,
        attrs=_freeze_attrs(attrs),
        operand_types=operand_types,
        result_types=result_types,
    )


def _validate_operation_shape(
    definition: OpDef,
    *,
    operands: tuple[str, ...],
    results: tuple[str, ...],
    attrs: Mapping[str, Any],
    name: str,
) -> None:
    ssa_operands = tuple(
        operand for operand in definition.operands if operand.kind == 'ssa'
    )
    min_operands = sum(not operand.optional for operand in ssa_operands)
    max_operands = len(ssa_operands)
    if not min_operands <= len(operands) <= max_operands:
        raise ValueError(
            f'{name} expects {min_operands}'
            + (f'..{max_operands}' if min_operands != max_operands else '')
            + f' SSA operands, got {len(operands)}.'
        )
    if len(results) != len(definition.results):
        raise ValueError(
            f'{name} expects {len(definition.results)} results, '
            f'got {len(results)}.'
        )

    required_attrs = {
        attr.name for attr in definition.attributes if attr.required
    }
    allowed_attrs = {attr.name for attr in definition.attributes} | {
        operand.name
        for operand in definition.operands
        if operand.kind == 'symbol'
    }
    missing = required_attrs - set(attrs)
    if missing:
        joined = ', '.join(sorted(missing))
        raise ValueError(f'{name} missing required attrs: {joined}.')
    unknown = set(attrs) - allowed_attrs
    if unknown:
        joined = ', '.join(sorted(unknown))
        raise ValueError(f'{name} has unknown attrs: {joined}.')
    for attr in definition.attributes:
        if attr.name in attrs:
            _validate_attr_value(
                name, attr.name, attr.kind, attrs[attr.name]
            )
    for operand in definition.operands:
        if operand.kind == 'symbol' and operand.name in attrs:
            _validate_attr_value(
                name, operand.name, 'str', attrs[operand.name]
            )


def _validate_attr_value(
    op_name: str,
    attr_name: str,
    kind: str,
    value: Any,
) -> None:
    if kind == 'packing':
        _validate_packing_attr(op_name, attr_name, value)
        return
    enum_values = _enum_attr_values().get(kind)
    if enum_values is not None:
        if not isinstance(value, str):
            raise TypeError(f'{op_name}.{attr_name} must be a string enum.')
        if value not in enum_values:
            joined = ', '.join(enum_values)
            raise ValueError(
                f'{op_name}.{attr_name} must be one of: {joined}.'
            )
        return
    validator = _ATTR_VALIDATORS.get(kind)
    if validator is not None:
        validator(op_name, attr_name, value)
        return
    raise ValueError(
        f'{op_name}.{attr_name} has unsupported attr kind {kind}.'
    )


def _validate_i64_triple(
    op_name: str,
    attr_name: str,
    value: Any,
) -> None:
    _validate_triple(op_name, attr_name, value, integer=True)


def _validate_f64_triple(
    op_name: str,
    attr_name: str,
    value: Any,
) -> None:
    _validate_triple(op_name, attr_name, value, integer=False)


def _validate_i64(op_name: str, attr_name: str, value: Any) -> None:
    if not _is_int(value):
        raise TypeError(f'{op_name}.{attr_name} must be an integer.')


def _validate_f32(op_name: str, attr_name: str, value: Any) -> None:
    if not _is_number(value):
        raise TypeError(f'{op_name}.{attr_name} must be numeric.')


def _validate_str(op_name: str, attr_name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f'{op_name}.{attr_name} must be a string.')


_ATTR_VALIDATORS = {
    'i64_triple': _validate_i64_triple,
    'f64_triple': _validate_f64_triple,
    'i64': _validate_i64,
    'f32': _validate_f32,
    'str': _validate_str,
}


def _validate_packing_attr(
    op_name: str,
    attr_name: str,
    value: Any,
) -> None:
    mapping = _mapping(value, f'{op_name}.{attr_name}')
    allowed = {'kind', 'group_size', 'scale_dtype', 'mode'}
    unknown = set(mapping) - allowed
    if unknown:
        joined = ', '.join(sorted(str(item) for item in unknown))
        raise ValueError(
            f'{op_name}.{attr_name} has unknown keys: {joined}.'
        )
    kind = mapping.get('kind')
    if kind not in _enum_attr_values()['packing']:
        raise ValueError(
            f'{op_name}.{attr_name}.kind must be dense, int4, or int8.'
        )
    if kind == 'dense':
        if 'group_size' in mapping and not _is_int(mapping['group_size']):
            raise TypeError(
                f'{op_name}.{attr_name}.group_size must be an integer.'
            )
        if 'scale_dtype' in mapping and mapping['scale_dtype'] not in (
            '',
            None,
        ):
            raise ValueError(
                f'{op_name}.{attr_name}.scale_dtype must be empty for dense.'
            )
        if 'mode' in mapping and mapping['mode'] not in ('', None):
            raise ValueError(
                f'{op_name}.{attr_name}.mode must be empty for dense.'
            )
        return

    group_size = mapping.get('group_size')
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size <= 0
    ):
        raise ValueError(
            f'{op_name}.{attr_name}.group_size must be a positive integer.'
        )
    if mapping.get('scale_dtype') not in ('f16', 'f32'):
        raise ValueError(
            f'{op_name}.{attr_name}.scale_dtype must be f16 or f32.'
        )
    if mapping.get('mode') != 'affine':
        raise ValueError(f'{op_name}.{attr_name}.mode must be affine.')


def _validate_triple(
    op_name: str,
    attr_name: str,
    value: Any,
    *,
    integer: bool,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f'{op_name}.{attr_name} must be a triple.')
    if len(value) != 3:
        raise ValueError(f'{op_name}.{attr_name} must contain 3 values.')
    predicate = _is_int if integer else _is_number
    if not all(predicate(item) for item in value):
        expected = 'integer' if integer else 'numeric'
        raise TypeError(
            f'{op_name}.{attr_name} must contain only {expected} values.'
        )


def _validate_plan_dataflow(
    *,
    ir_version: int,
    schema_digest: str,
    weight_file: str,
    name: str,
    args: tuple[PlanArgument, ...],
    ops: tuple[PlanOperation, ...],
    returns: tuple[str, ...],
    outputs: tuple[PlanOutput, ...],
) -> None:
    if ir_version != CURRENT_DIALECT_VERSION:
        raise ValueError(
            'unsupported lattice artifact runtime plan ir_version: '
            f'{ir_version} (expected {CURRENT_DIALECT_VERSION}).'
        )
    if schema_digest != DIALECT_SCHEMA_DIGEST:
        raise ValueError(
            'unsupported lattice artifact runtime plan schema_digest: '
            f'{schema_digest!r} (expected {DIALECT_SCHEMA_DIGEST!r}).'
        )
    if weight_file != ARTIFACT_WEIGHT_FILE:
        raise ValueError(
            'unsupported lattice artifact runtime plan weight_file: '
            f'{weight_file!r} (expected {ARTIFACT_WEIGHT_FILE!r}).'
        )
    if name != 'forward':
        raise ValueError(
            'lattice artifact runtime plan entry must be forward.'
        )
    if not returns:
        raise ValueError(
            'lattice artifact runtime plan must return values.'
        )

    defined: set[str] = set()
    input_names: set[str] = set()
    for argument in args:
        _validate_value_label(argument.name, 'argument')
        _validate_abi_name(argument.abi_name, 'argument')
        _validate_input_role(argument.role)
        if argument.name in defined:
            raise ValueError(
                f'duplicate runtime value label: {argument.name}'
            )
        if argument.abi_name in input_names:
            raise ValueError(
                f'duplicate artifact input ABI name: {argument.abi_name}'
            )
        defined.add(argument.name)
        input_names.add(argument.abi_name)

    for operation in ops:
        for operand in operation.operands:
            if operand not in defined:
                raise ValueError(
                    f'{operation.name} uses undefined value: {operand}'
                )
        for result in operation.results:
            _validate_value_label(result, f'{operation.name} result')
            if result in defined:
                raise ValueError(f'duplicate runtime value label: {result}')
            defined.add(result)

    for value in returns:
        if value not in defined:
            raise ValueError(f'return uses undefined value: {value}')

    if len(outputs) != len(returns):
        raise ValueError(
            'lattice artifact runtime plan outputs must match returns.'
        )
    output_names: set[str] = set()
    for output, value in zip(outputs, returns, strict=True):
        if output.name != value:
            raise ValueError(
                'lattice artifact runtime plan output names must match '
                'returns.'
            )
        _validate_abi_name(output.abi_name, 'output')
        _validate_output_role(output.role)
        if output.abi_name in output_names:
            raise ValueError(
                f'duplicate artifact output ABI name: {output.abi_name}'
            )
        output_names.add(output.abi_name)


def _validate_value_label(value: str, label: str) -> None:
    if not value:
        raise ValueError(f'{label} value label must not be empty.')


def _validate_abi_name(value: str, label: str) -> None:
    if not value:
        raise ValueError(f'{label} ABI name must not be empty.')


def _validate_input_role(value: str) -> None:
    if value not in {
        'tensor',
        'sparse_coords',
        'sparse_features',
        'sparse_active',
    }:
        raise ValueError(f'unsupported artifact input ABI role: {value}.')


def _validate_output_role(value: str) -> None:
    if value not in {'tensor', 'sparse_tensor'}:
        raise ValueError(f'unsupported artifact output ABI role: {value}.')


def _enum_attr_values() -> Mapping[str, tuple[str, ...]]:
    return {
        definition.mnemonic: definition.values
        for definition in LATTICE_DIALECT.attrs.values()
        if definition.values
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _freeze_attrs(attrs: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _freeze_attr_value(value)
            for key, value in attrs.items()
        }
    )


def _freeze_attr_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_attr_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(_freeze_attr_value(item) for item in value)
    return value


def _optional_str_tuple(
    mapping: Mapping[str, Any], name: str
) -> tuple[str, ...]:
    if name not in mapping:
        return ()
    return tuple(str(item) for item in _sequence(mapping, name))


def _str(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping[name]
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string.')
    return value


def _int(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping[name]
    if not isinstance(value, int):
        raise TypeError(f'{name} must be an integer.')
    return value


def _sequence(mapping: Mapping[str, Any], name: str) -> Sequence[Any]:
    value = mapping[name]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f'{name} must be a sequence.')
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping.')
    return value

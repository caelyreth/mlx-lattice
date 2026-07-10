from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from lattice_contract.dialect import sparse_decompose, sparse_make, weight

from mlx_lattice import _ext as ext
from mlx_lattice.core import SparseTensor

from .io import LatticeArtifact, load_lattice_artifact
from .lowering import (
    ARTIFACT_LOWERINGS,
    LoweringProgram,
    PackedWeightPayload,
    RuntimeValue,
    artifact_lowering,
    int_attr,
    packing_kind,
    results,
)
from .plan import PlanArgument, PlanOperation, RuntimePlan


@dataclass(frozen=True, slots=True)
class LatticeProgram:
    """Executable MLX lowering of a verified lattice MLIR artifact.

    The plan is produced by the MLIR-enabled native extension after parsing and
    verifying the graph with the registered ``lattice`` dialect. Python only
    lowers the structured plan to existing ``mlx_lattice.ops`` calls.
    """

    plan: RuntimePlan
    weights: Mapping[str, mx.array]

    @classmethod
    def from_artifact(cls, artifact: LatticeArtifact) -> LatticeProgram:
        """Compile a loaded artifact through the native MLIR importer."""

        _register_artifact_lowerings()
        native = _require_native_artifact_execution()
        return cls(
            RuntimePlan.from_native(
                cast(Mapping[str, Any], native(artifact.graph))
            ),
            artifact.weights,
        )

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> RuntimeValue | tuple[RuntimeValue, ...]:
        """Execute the lowered program with positional or named ABI inputs."""

        values = self._bind_inputs(args, kwargs)
        for operation in self.plan.ops:
            result = ARTIFACT_LOWERINGS.lower(self, operation, values)
            _store_results(operation, values, result)
        returns = tuple(values[name] for name in self.plan.returns)
        return returns[0] if len(returns) == 1 else returns

    def _bind_inputs(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, RuntimeValue]:
        plan_args = self.plan.args
        if any(
            isinstance(value, SparseTensor)
            for value in (*args, *kwargs.values())
        ):
            return _bind_logical_inputs(plan_args, args, kwargs)
        value_names = tuple(item.name for item in plan_args)
        abi_names = tuple(item.abi_name for item in plan_args)
        if len(args) > len(value_names):
            raise ValueError('too many positional artifact inputs.')
        values: dict[str, RuntimeValue] = {}
        for plan_arg, value in zip(plan_args, args, strict=False):
            values[plan_arg.name] = _runtime_value(value)
        for plan_arg in plan_args[len(args) :]:
            if plan_arg.abi_name not in kwargs:
                raise ValueError(
                    f'missing artifact input: {plan_arg.abi_name}'
                )
            values[plan_arg.name] = _runtime_value(
                kwargs[plan_arg.abi_name]
            )
        unexpected = set(kwargs) - set(abi_names)
        if unexpected:
            names = ', '.join(sorted(unexpected))
            raise ValueError(f'unexpected artifact inputs: {names}')
        return values


def compile_lattice_artifact(artifact: LatticeArtifact) -> LatticeProgram:
    """Compile a loaded artifact into an executable MLX program."""

    return LatticeProgram.from_artifact(artifact)


def load_lattice_program(path: str | Path) -> LatticeProgram:
    """Load an artifact directory and compile it into an MLX program."""

    return compile_lattice_artifact(load_lattice_artifact(path))


def native_artifact_execution_available() -> bool:
    """Return whether this install can execute MLIR artifacts natively."""

    return callable(getattr(ext, 'lattice_mlir_plan', None))


def _require_native_artifact_execution():
    native = getattr(ext, 'lattice_mlir_plan', None)
    if callable(native):
        return native
    raise RuntimeError(
        'artifact execution requires an MLIR-enabled mlx-lattice native '
        'extension. The MLIR artifact contract is still available for bundle '
        'IO and validation, but this installed extension was built without '
        'native artifact execution support.'
    )


def _register_artifact_lowerings() -> None:
    """Import operation modules that declare artifact lowering annotations."""

    for module in (
        'mlx_lattice.ops.conv',
        'mlx_lattice.ops.feature',
        'mlx_lattice.ops.pool',
        'mlx_lattice.ops.quantization',
        'mlx_lattice.ops.tensor',
    ):
        import_module(module)


@artifact_lowering(op=weight)
def weight_from_artifact(
    program: LoweringProgram,
    storage_key: str,
    layout: str,
    packing: Mapping[str, Any],
) -> mx.array | PackedWeightPayload:
    if not isinstance(packing, Mapping):
        raise TypeError('weight packing must be a mapping.')
    if packing.get('kind') == 'dense':
        try:
            return program.weights[storage_key]
        except KeyError as exc:
            raise ValueError(
                f'artifact weight not found: {storage_key}'
            ) from exc

    kind = packing_kind(packing.get('kind'))
    group_size = int_attr(packing, 'group_size')
    scale_dtype = str(packing.get('scale_dtype', ''))
    mode = str(packing.get('mode', ''))
    try:
        packed_weight = program.weights[f'{storage_key}.weight']
        scales = program.weights[f'{storage_key}.scales']
        biases = program.weights[f'{storage_key}.biases']
    except KeyError as exc:
        raise ValueError(
            'quantized artifact weight requires '
            f'{storage_key}.weight, {storage_key}.scales, and '
            f'{storage_key}.biases tensors.'
        ) from exc
    return PackedWeightPayload(
        storage_key=storage_key,
        layout=layout,
        kind=kind,
        group_size=group_size,
        scale_dtype=scale_dtype,
        mode=mode,
        weight=packed_weight,
        scales=scales,
        biases=biases,
    )


@artifact_lowering(op=sparse_make)
def sparse_make_from_artifact(
    coords: mx.array,
    features: mx.array,
    active: mx.array,
    *,
    stride: tuple[int, int, int],
    coord_order: str,
) -> SparseTensor:
    if coord_order != 'batch_x_y_z':
        raise ValueError(
            f'unsupported sparse coordinate order: {coord_order}'
        )
    return SparseTensor(
        coords,
        features,
        stride=stride,
        active_rows=active,
    )


@artifact_lowering(op=sparse_decompose)
def sparse_decompose_from_artifact(
    input: SparseTensor,
) -> tuple[mx.array, mx.array, mx.array]:
    return (input.coords, input.feats, input.active_rows)


def _store_results(
    operation: PlanOperation,
    values: dict[str, RuntimeValue],
    result: RuntimeValue | tuple[RuntimeValue, ...],
) -> None:
    op_results = results(operation)
    values_to_store = (
        cast(tuple[RuntimeValue, ...], result)
        if isinstance(result, tuple)
        else (result,)
    )
    if len(op_results) != len(values_to_store):
        raise ValueError(
            f'{operation.name} produced {len(values_to_store)} results, '
            f'expected {len(op_results)}.'
        )
    for name, value in zip(op_results, values_to_store, strict=True):
        values[name] = value


def _runtime_value(value: Any) -> RuntimeValue:
    if isinstance(value, SparseTensor | mx.array):
        return value
    raise TypeError('artifact inputs must be SparseTensor or MLX arrays.')


@dataclass(frozen=True, slots=True)
class _InputGroup:
    abi_name: str
    arguments: tuple[PlanArgument, ...]


def _bind_logical_inputs(
    plan_args: Sequence[PlanArgument],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, RuntimeValue]:
    groups = _input_groups(plan_args)
    component_names = {argument.abi_name for argument in plan_args}
    if any(isinstance(value, SparseTensor) for value in args) and (
        set(kwargs) & component_names
    ):
        raise ValueError(
            'SparseTensor shorthand cannot be combined with component '
            'artifact inputs.'
        )
    if len(args) > len(groups):
        raise ValueError('too many positional artifact inputs.')
    values: dict[str, RuntimeValue] = {}
    used: set[str] = set()
    for group, value in zip(groups, args, strict=False):
        _bind_input_group(values, group, value)
    for group in groups[len(args) :]:
        if group.abi_name not in kwargs:
            if any(
                argument.abi_name in kwargs for argument in group.arguments
            ):
                raise ValueError(
                    'SparseTensor shorthand cannot be combined with '
                    'component artifact inputs.'
                )
            raise ValueError(f'missing artifact input: {group.abi_name}')
        _bind_input_group(values, group, kwargs[group.abi_name])
        used.add(group.abi_name)
    unexpected = set(kwargs) - used
    if unexpected:
        names = ', '.join(sorted(unexpected))
        raise ValueError(f'unexpected artifact inputs: {names}')
    return values


def _input_groups(
    plan_args: Sequence[PlanArgument],
) -> tuple[_InputGroup, ...]:
    groups: list[_InputGroup] = []
    index = 0
    sparse_roles = ('sparse_coords', 'sparse_features', 'sparse_active')
    while index < len(plan_args):
        argument = plan_args[index]
        if argument.role != 'sparse_coords':
            groups.append(_InputGroup(argument.abi_name, (argument,)))
            index += 1
            continue
        sparse = tuple(plan_args[index : index + 3])
        if (
            len(sparse) != 3
            or tuple(item.role for item in sparse) != sparse_roles
        ):
            raise ValueError(
                'SparseTensor shorthand requires consecutive sparse_coords, '
                'sparse_features, and sparse_active ABI arguments.'
            )
        groups.append(_InputGroup(_sparse_abi_name(sparse), sparse))
        index += 3
    return tuple(groups)


def _sparse_abi_name(arguments: tuple[PlanArgument, ...]) -> str:
    suffixes = ('_coords', '_features', '_active')
    names = tuple(argument.abi_name for argument in arguments)
    if names == ('coords', 'features', 'active'):
        return 'input'
    prefixes = tuple(
        name[: -len(suffix)] if name.endswith(suffix) else ''
        for name, suffix in zip(names, suffixes, strict=True)
    )
    if not prefixes[0] or len(set(prefixes)) != 1:
        raise ValueError(
            'sparse artifact ABI names must share the '
            '<input>_coords/features/active prefix.'
        )
    return prefixes[0]


def _bind_input_group(
    values: dict[str, RuntimeValue],
    group: _InputGroup,
    value: Any,
) -> None:
    if len(group.arguments) == 1:
        if isinstance(value, SparseTensor):
            raise ValueError(
                'SparseTensor shorthand requires sparse_coords, '
                'sparse_features, and sparse_active ABI roles.'
            )
        values[group.arguments[0].name] = _runtime_value(value)
        return
    if not isinstance(value, SparseTensor):
        raise TypeError(
            f'artifact input {group.abi_name!r} must be a SparseTensor.'
        )
    runtime_values = (value.coords, value.feats, value.active_rows)
    for argument, runtime_value in zip(
        group.arguments, runtime_values, strict=True
    ):
        values[argument.name] = runtime_value

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
        native = getattr(ext, 'lattice_mlir_plan', None)
        if native is None:
            raise RuntimeError(
                'artifact execution requires an MLIR-enabled mlx-lattice '
                'native extension.'
            )
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
        value_names = tuple(item.name for item in plan_args)
        abi_names = tuple(item.abi_name for item in plan_args)
        if len(args) == 1 and isinstance(args[0], SparseTensor):
            if kwargs:
                raise ValueError(
                    'SparseTensor shorthand cannot be combined with keyword '
                    'artifact inputs.'
                )
            coords, features, active = _sparse_tensor_abi_args(plan_args)
            return {
                coords: args[0].coords,
                features: args[0].feats,
                active: args[0].active_rows,
            }

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


def _sparse_tensor_abi_args(
    plan_args: Sequence[PlanArgument],
) -> tuple[str, str, str]:
    if len(plan_args) < 3:
        raise ValueError(
            'SparseTensor shorthand requires coords/features/active artifact '
            'ABI arguments.'
        )
    expected_roles = ('sparse_coords', 'sparse_features', 'sparse_active')
    actual_roles = tuple(item.role for item in plan_args[:3])
    if actual_roles != expected_roles:
        raise ValueError(
            'SparseTensor shorthand requires the first three artifact ABI '
            'arguments to be tagged as sparse_coords, sparse_features, and '
            'sparse_active by the native MLIR importer.'
        )
    return (
        plan_args[0].name,
        plan_args[1].name,
        plan_args[2].name,
    )

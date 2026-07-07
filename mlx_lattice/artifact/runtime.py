from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    RuntimeValue,
    array,
    artifact_lowering,
    attrs,
    operands,
    results,
    sparse,
    str_attr,
    triple_attr,
)


@dataclass(frozen=True, slots=True)
class LatticeProgram:
    """Executable MLX lowering of a verified lattice MLIR artifact.

    The plan is produced by the MLIR-enabled native extension after parsing and
    verifying the graph with the registered ``lattice`` dialect. Python only
    lowers the structured plan to existing ``mlx_lattice.ops`` calls.
    """

    plan: Mapping[str, Any]
    weights: Mapping[str, mx.array]

    @classmethod
    def from_artifact(cls, artifact: LatticeArtifact) -> LatticeProgram:
        """Compile a loaded artifact through the native MLIR importer."""

        _register_feature_lowerings()
        native = getattr(ext, 'lattice_mlir_plan', None)
        if native is None:
            raise RuntimeError(
                'artifact execution requires an MLIR-enabled mlx-lattice '
                'native extension.'
            )
        return cls(
            cast(Mapping[str, Any], native(artifact.graph)),
            artifact.weights,
        )

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> RuntimeValue | tuple[RuntimeValue, ...]:
        """Execute the lowered program with positional or named ABI inputs."""

        values = self._bind_inputs(args, kwargs)
        for operation in cast(
            Sequence[Mapping[str, Any]], self.plan['ops']
        ):
            result = ARTIFACT_LOWERINGS.lower(self, operation, values)
            _store_results(operation, values, result)
        returns = tuple(
            values[str(name)]
            for name in cast(Sequence[Any], self.plan['returns'])
        )
        return returns[0] if len(returns) == 1 else returns

    def _bind_inputs(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, RuntimeValue]:
        plan_args = cast(Sequence[Mapping[str, Any]], self.plan['args'])
        arg_names = tuple(str(item['name']) for item in plan_args)
        if len(args) == 1 and isinstance(args[0], SparseTensor):
            if arg_names[:3] != ('arg0', 'arg1', 'arg2'):
                raise ValueError(
                    'SparseTensor shorthand requires coords/features/active '
                    'as the first three artifact ABI arguments.'
                )
            if kwargs:
                raise ValueError(
                    'SparseTensor shorthand cannot be combined with keyword '
                    'artifact inputs.'
                )
            return {
                'arg0': args[0].coords,
                'arg1': args[0].feats,
                'arg2': args[0].active_rows,
            }

        if len(args) > len(arg_names):
            raise ValueError('too many positional artifact inputs.')
        values: dict[str, RuntimeValue] = {}
        for name, value in zip(arg_names, args, strict=False):
            values[name] = _runtime_value(value)
        for name in arg_names[len(args) :]:
            if name not in kwargs:
                raise ValueError(f'missing artifact input: {name}')
            values[name] = _runtime_value(kwargs[name])
        unexpected = set(kwargs) - set(arg_names)
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


def _register_feature_lowerings() -> None:
    """Import feature modules that declare artifact lowering annotations."""

    from mlx_lattice.ops import conv as _conv
    from mlx_lattice.ops import feature as _feature
    from mlx_lattice.ops import tensor as _tensor

    del _conv, _feature, _tensor


@artifact_lowering(op=weight)
def weight_from_artifact(
    program: LoweringProgram,
    operation: Mapping[str, Any],
    values: dict[str, RuntimeValue],
) -> mx.array:
    del values
    op_attrs = attrs(operation)
    storage_key = str_attr(op_attrs, 'storage_key')
    packing = op_attrs.get('packing', {'kind': 'dense'})
    if not isinstance(packing, Mapping) or packing.get('kind') != 'dense':
        raise ValueError(
            'artifact execution currently supports dense weights only.'
        )
    try:
        return program.weights[storage_key]
    except KeyError as exc:
        raise ValueError(
            f'artifact weight not found: {storage_key}'
        ) from exc


@artifact_lowering(op=sparse_make)
def sparse_make_from_artifact(
    program: LoweringProgram,
    operation: Mapping[str, Any],
    values: dict[str, RuntimeValue],
) -> SparseTensor:
    del program
    op_operands = operands(operation)
    op_attrs = attrs(operation)
    coord_order = str_attr(op_attrs, 'coord_order')
    if coord_order != 'batch_x_y_z':
        raise ValueError(
            f'unsupported sparse coordinate order: {coord_order}'
        )
    return SparseTensor(
        array(values, op_operands[0]),
        array(values, op_operands[1]),
        stride=triple_attr(op_attrs, 'stride'),
        active_rows=array(values, op_operands[2]),
    )


@artifact_lowering(op=sparse_decompose)
def sparse_decompose_from_artifact(
    program: LoweringProgram,
    operation: Mapping[str, Any],
    values: dict[str, RuntimeValue],
) -> tuple[mx.array, mx.array, mx.array]:
    del program
    value = sparse(values, operands(operation)[0])
    return (value.coords, value.feats, value.active_rows)


def _store_results(
    operation: Mapping[str, Any],
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
            f'{operation["name"]} produced {len(values_to_store)} results, '
            f'expected {len(op_results)}.'
        )
    for name, value in zip(op_results, values_to_store, strict=True):
        values[name] = value


def _runtime_value(value: Any) -> RuntimeValue:
    if isinstance(value, SparseTensor | mx.array):
        return value
    raise TypeError('artifact inputs must be SparseTensor or MLX arrays.')

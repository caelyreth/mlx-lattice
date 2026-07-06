from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal, Protocol, cast

import mlx.core as mx
from lattice_contract import (
    FEATURE_BATCH_NORM,
    FEATURE_DROPOUT,
    FEATURE_GELU,
    FEATURE_LAYER_NORM,
    FEATURE_LEAKY_RELU,
    FEATURE_LINEAR,
    FEATURE_RELU,
    FEATURE_RMS_NORM,
    FEATURE_SIGMOID,
    FEATURE_SILU,
    FEATURE_SOFTPLUS,
    FEATURE_TANH,
    POOL3D,
    POOL_AVG3D,
    POOL_GLOBAL_AVG,
    POOL_GLOBAL_MAX,
    POOL_GLOBAL_SUM,
    POOL_MAX3D,
    POOL_SUM3D,
    SPARSE_ADD,
    SPARSE_CONV3D,
    SPARSE_CONV_TRANSPOSE3D,
    SPARSE_GENERATIVE_CONV_TRANSPOSE3D,
    SPARSE_SUBM_CONV3D,
    VALUE_FIELD,
    IRNode,
    IRValueType,
)
from lattice_contract.manifest import IRInputRef

import mlx_lattice.ops as ops
from mlx_lattice.artifact.bindings import (
    ExecutionContext,
    GraphValue,
    iter_value_type_bindings,
    value_type_fields,
)
from mlx_lattice.core import QuantizedWeight, SparseTensor
from mlx_lattice.ops._quantized import quantized_matmul


class OperationRegistrar(Protocol):
    """Callable registry interface used by operation registration modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> Callable: ...

    def binding(self, name: str): ...


def register_operations(lattice_op: OperationRegistrar) -> None:
    """Register the legacy JSON artifact's explicit semantic op set."""

    register_artifact_ops(lattice_op)
    register_semantic_ops(lattice_op)


def register_artifact_ops(lattice_op: OperationRegistrar) -> None:
    """Register small artifact-graph utility ops."""

    @lattice_op(
        VALUE_FIELD,
        function=_field_value,
        inputs={'input': 'value'},
    )
    def _field(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        value = context.input_value(node.inputs['input'])
        field = node.attributes.get('field')
        if not isinstance(field, str):
            raise ValueError(
                f'{node.id}.attributes.field must be a string.'
            )
        return {'output': _field_value(value, field)}


def register_semantic_ops(lattice_op: OperationRegistrar) -> None:
    """Register stable semantic op aliases used by persisted artifacts."""

    p = public_ops()

    for contract, fn in (
        (
            SPARSE_CONV3D,
            p['conv3d'],
        ),
        (
            SPARSE_SUBM_CONV3D,
            p['subm_conv3d'],
        ),
        (
            SPARSE_CONV_TRANSPOSE3D,
            p['conv_transpose3d'],
        ),
        (
            SPARSE_GENERATIVE_CONV_TRANSPOSE3D,
            p['generative_conv_transpose3d'],
        ),
    ):

        @lattice_op(
            contract,
            function=fn,
            inputs={'input': 'x'},
        )
        def _conv(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)

    @lattice_op(
        SPARSE_ADD,
        function=p['sparse_add'],
        inputs={'lhs': 'lhs', 'rhs': 'rhs'},
    )
    def _sparse_add(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        return passthrough(lattice_op, context, node)

    @lattice_op(
        FEATURE_LINEAR,
        function=p['linear'],
        inputs={'input': 'x'},
        handler=_linear(lattice_op),
    )
    def _linear_op(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        return passthrough(lattice_op, context, node)

    for contract, name in (
        (FEATURE_RELU, 'relu'),
        (FEATURE_SIGMOID, 'sigmoid'),
        (FEATURE_SILU, 'silu'),
        (FEATURE_TANH, 'tanh'),
        (FEATURE_GELU, 'gelu'),
        (FEATURE_LEAKY_RELU, 'leaky_relu'),
        (FEATURE_SOFTPLUS, 'softplus'),
        (FEATURE_DROPOUT, 'dropout'),
        (FEATURE_BATCH_NORM, 'batch_norm'),
        (FEATURE_LAYER_NORM, 'layer_norm'),
        (FEATURE_RMS_NORM, 'rms_norm'),
    ):

        @lattice_op(
            contract,
            function=p[name],
            inputs={'input': 'x'},
        )
        def _feature(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)

    for contract, name in (
        (POOL3D, 'pool3d'),
        (POOL_SUM3D, 'sum_pool3d'),
        (POOL_MAX3D, 'max_pool3d'),
        (POOL_AVG3D, 'avg_pool3d'),
    ):

        @lattice_op(
            contract,
            function=p[name],
            inputs={'input': 'x'},
        )
        def _local_pool(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)

    for contract, name in (
        (POOL_GLOBAL_SUM, 'sum'),
        (POOL_GLOBAL_AVG, 'avg'),
        (POOL_GLOBAL_MAX, 'max'),
    ):

        @lattice_op(
            contract,
            function=p[f'global_{name}_pool'],
            inputs={'input': 'x'},
            handler=_pool_global(lattice_op),
        )
        def _global(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)


def public_ops() -> dict[str, Callable[..., GraphValue]]:
    """Return public functional ops for diagnostics only."""

    return {
        name: cast(Callable[..., GraphValue], getattr(ops, name))
        for name in ops.__all__
        if inspect.isfunction(getattr(ops, name))
    }


def field_value_type(value_type: IRValueType, field: str) -> IRValueType:
    """Return the output value type for a supported structural field."""

    fields = value_type_fields(value_type)
    if field not in fields:
        raise ValueError(
            f'field {field!r} is not supported for IR value type '
            f'{value_type!r}.'
        )
    return fields[field]


def passthrough(
    lattice_op: OperationRegistrar,
    context: ExecutionContext,
    node: IRNode,
) -> dict[str, GraphValue]:
    return lattice_op.binding(node.op).run_default(context, node)


def _linear(lattice_op: OperationRegistrar):
    def handler(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        binding = lattice_op.binding(node.op)
        kwargs = binding.arguments(context, node)
        x = kwargs['x']
        if isinstance(x, mx.array):
            return {
                binding.output: _dense_linear(
                    x,
                    kwargs['weight'],
                    cast(mx.array | None, kwargs.get('bias')),
                )
            }
        return {binding.output: binding.function(**kwargs)}

    return handler


def _dense_linear(
    x: mx.array,
    weight: mx.array | QuantizedWeight,
    bias: mx.array | None,
) -> mx.array:
    if isinstance(weight, QuantizedWeight):
        out = quantized_matmul(x, weight)
    else:
        if weight.ndim != 2:
            raise ValueError('weight must have shape (C_out, C_in).')
        if x.ndim < 1 or int(x.shape[-1]) != int(weight.shape[1]):
            raise ValueError(
                'weight input channels must match x trailing dimension.'
            )
        out = x @ weight.T
    if bias is None:
        return out
    if bias.ndim != 1 or int(bias.shape[0]) != int(out.shape[-1]):
        raise ValueError('bias must have shape (C_out,).')
    return out + bias


def _field_value(value: GraphValue, field: str) -> GraphValue:
    if not _is_allowed_field(value, field):
        raise ValueError(
            f'field {field!r} is not supported for {type(value).__name__}.'
        )
    return cast(GraphValue, getattr(value, field))


def _is_allowed_field(value: GraphValue, field: str) -> bool:
    for binding in iter_value_type_bindings():
        if (
            binding.fields
            and isinstance(value, binding.runtime_type)
            and field in binding.fields
        ):
            return True
    return False


def _pool_global(lattice_op: OperationRegistrar):
    def handler(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        x = context.sparse(_str_ref(node.inputs['input']))
        binding = lattice_op.binding(node.op)
        fn = cast(Callable[[SparseTensor], mx.array], binding.function)
        if x.batch_counts is not None or node.op == POOL_GLOBAL_MAX.name:
            return {'output': fn(x)}
        if context.batch_size is None:
            raise ValueError(
                f'{node.id} requires batch_counts metadata or a sparse graph '
                'input with known batch_counts.'
            )
        mode = node.op.removeprefix('pool.global_')
        return {
            'output': _global_pool_from_coordinate_batches(
                x,
                context.batch_size,
                mode=cast(Any, mode),
            )
        }

    return handler


def _global_pool_from_coordinate_batches(
    x: SparseTensor,
    batch_size: int,
    *,
    mode: Literal['sum', 'avg'],
) -> mx.array:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    rows = mx.arange(x.capacity, dtype=mx.int32)
    active = (rows < x.active_rows[0]).astype(x.feats.dtype)
    batch_ids = x.coords[:, 0].astype(mx.int32)
    clipped = mx.minimum(mx.maximum(batch_ids, 0), batch_size - 1)
    summed = (
        mx.zeros((batch_size, x.channels), dtype=x.feats.dtype)
        .at[clipped]
        .add(x.feats * active[:, None])
    )
    if mode == 'sum':
        return summed
    counts = (
        mx.zeros((batch_size,), dtype=x.feats.dtype).at[clipped].add(active)
    )
    return summed / mx.maximum(counts, 1)[:, None]


def _str_ref(value: IRInputRef) -> str:
    if not isinstance(value, str):
        raise ValueError('expected a single graph value reference.')
    return value

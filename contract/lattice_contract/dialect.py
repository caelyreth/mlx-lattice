from __future__ import annotations

from lattice_contract.schema import (
    DialectSchema,
    attr_param,
    op_attr,
    operand,
    result,
    type_param,
)

LATTICE_DIALECT = DialectSchema('lattice')


@LATTICE_DIALECT.type(
    'SparseTensor',
    'sparse_tensor',
    parameters=(
        type_param('rank', 'unsigned'),
        type_param('coord', 'string'),
        type_param('feature', 'string'),
        type_param('dtype', 'type'),
    ),
    summary='Portable sparse lattice value',
)
class SparseTensorType:
    """Annotated sparse tensor type declaration."""


@LATTICE_DIALECT.type(
    'Weight',
    'weight',
    parameters=(
        type_param('family', 'string'),
        type_param('dtype', 'type'),
    ),
    summary='Symbolic model weight',
)
class WeightType:
    """Annotated symbolic weight type declaration."""


@LATTICE_DIALECT.attr(
    'Coord',
    'coord',
    parameters=(attr_param('value', 'string'),),
    values=('batch_x_y_z',),
    summary='Sparse coordinate convention',
)
class CoordAttr:
    """Annotated coordinate convention attribute."""


@LATTICE_DIALECT.attr(
    'FeatureLayout',
    'feature_layout',
    parameters=(attr_param('value', 'string'),),
    values=('row_channel',),
    summary='Sparse feature matrix layout',
)
class FeatureLayoutAttr:
    """Annotated feature layout attribute."""


@LATTICE_DIALECT.attr(
    'WeightLayout',
    'weight_layout',
    parameters=(attr_param('value', 'string'),),
    values=('conv3d_o_zyx_i', 'linear_o_i'),
    summary='Logical weight layout',
)
class WeightLayoutAttr:
    """Annotated weight layout attribute."""


@LATTICE_DIALECT.attr(
    'Packing',
    'packing',
    parameters=(
        attr_param('kind', 'string'),
        attr_param('group_size', 'unsigned'),
        attr_param('scale_type', 'type'),
        attr_param('mode', 'string'),
    ),
    values=('dense', 'int4', 'int8'),
    summary='Weight storage packing',
)
class PackingAttr:
    """Annotated weight packing attribute."""


@LATTICE_DIALECT.attr(
    'Join',
    'join',
    parameters=(attr_param('value', 'string'),),
    values=('inner', 'left', 'right', 'outer'),
    summary='Coordinate-aligned sparse algebra join mode',
)
class JoinAttr:
    """Annotated sparse algebra join attribute."""


@LATTICE_DIALECT.op(
    'weight',
    operands=(operand('sym_name', 'symbol', kind='symbol'),),
    results=(result('result', 'weight'),),
    attributes=(
        op_attr('storage_key', 'str'),
        op_attr('layout', 'weight_layout'),
        op_attr('packing', 'packing'),
    ),
    assembly='weight',
    summary='Bind a symbolic weight to external tensor storage',
)
def weight() -> None:
    """Register lattice.weight."""


@LATTICE_DIALECT.op(
    'sparse.make',
    operands=(
        operand('coords', 'tensor'),
        operand('features', 'tensor'),
        operand('active', 'tensor'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('stride', 'i64_triple'),
        op_attr('coord_order', 'coord'),
    ),
    assembly='functional',
    summary='Construct a sparse value from ABI tensor components',
)
def sparse_make() -> None:
    """Register lattice.sparse.make."""


@LATTICE_DIALECT.op(
    'sparse.decompose',
    operands=(operand('input', 'sparse_tensor'),),
    results=(
        result('coords', 'tensor'),
        result('features', 'tensor'),
        result('active', 'tensor'),
    ),
    assembly='decompose',
    summary='Expose sparse value ABI tensor components',
)
def sparse_decompose() -> None:
    """Register lattice.sparse.decompose."""


@LATTICE_DIALECT.op(
    'sparse.with_features',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('features', 'tensor'),
    ),
    results=(result('result', 'sparse_tensor'),),
    assembly='functional',
    summary='Replace sparse features while preserving coordinate support',
)
def sparse_with_features() -> None:
    """Register lattice.sparse.with_features."""


_CONV_ATTRS = (
    op_attr('kernel_size', 'i64_triple'),
    op_attr('stride', 'i64_triple'),
    op_attr('padding', 'i64_triple'),
    op_attr('dilation', 'i64_triple'),
)


@LATTICE_DIALECT.op(
    'conv3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('weight', 'weight'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=_CONV_ATTRS,
    assembly='functional',
    summary='Forward sparse 3D convolution',
)
def conv3d() -> None:
    """Register lattice.conv3d."""


@LATTICE_DIALECT.op(
    'subm_conv3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('weight', 'weight'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('kernel_size', 'i64_triple'),
        op_attr('dilation', 'i64_triple'),
    ),
    assembly='functional',
    summary='Submanifold sparse 3D convolution',
)
def subm_conv3d() -> None:
    """Register lattice.subm_conv3d."""


@LATTICE_DIALECT.op(
    'target_conv3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('target', 'sparse_tensor'),
        operand('weight', 'weight'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=_CONV_ATTRS,
    assembly='functional',
    summary='Sparse 3D convolution on explicit target coordinates',
)
def target_conv3d() -> None:
    """Register lattice.target_conv3d."""


@LATTICE_DIALECT.op(
    'linear',
    operands=(
        operand('input', 'tensor'),
        operand('weight', 'weight'),
    ),
    results=(result('result', 'tensor'),),
    assembly='functional',
    summary='Dense linear projection over feature tensors',
)
def linear() -> None:
    """Register lattice.linear."""


@LATTICE_DIALECT.op(
    'sparse.add',
    operands=(
        operand('lhs', 'sparse_tensor'),
        operand('rhs', 'sparse_tensor'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('join', 'join'),
        op_attr('lhs_fill', 'f32'),
        op_attr('rhs_fill', 'f32'),
    ),
    assembly='functional',
    summary='Coordinate-aligned sparse addition',
)
def sparse_add() -> None:
    """Register lattice.sparse.add."""

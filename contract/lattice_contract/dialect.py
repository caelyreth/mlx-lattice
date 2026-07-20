from __future__ import annotations

from lattice_contract.kernel import CANONICAL_CONV3D_WEIGHT_LAYOUT
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
    values=(
        CANONICAL_CONV3D_WEIGHT_LAYOUT,
        'linear_o_i',
        'embedding_n_c',
        'channel_c',
        'bias_c',
    ),
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
        attr_param('scale_dtype', 'type'),
        attr_param('mode', 'string'),
    ),
    values=('dense', 'int4', 'int8'),
    summary='Weight storage packing',
)
class PackingAttr:
    """Annotated weight packing attribute."""


@LATTICE_DIALECT.attr(
    'Activation',
    'activation',
    parameters=(attr_param('value', 'string'),),
    values=(
        'relu',
        'sigmoid',
        'gelu',
        'silu',
        'leaky_relu',
        'tanh',
        'softplus',
    ),
    summary='Dense feature activation function',
)
class ActivationAttr:
    """Annotated dense feature activation attribute."""


@LATTICE_DIALECT.attr(
    'GeluApprox',
    'gelu_approx',
    parameters=(attr_param('value', 'string'),),
    values=('none', 'precise', 'tanh', 'fast'),
    summary='GELU approximation mode',
)
class GeluApproxAttr:
    """Annotated GELU approximation attribute."""


@LATTICE_DIALECT.attr(
    'Join',
    'join',
    parameters=(attr_param('value', 'string'),),
    values=('inner', 'left', 'right', 'outer'),
    summary='Coordinate-aligned sparse algebra join mode',
)
class JoinAttr:
    """Annotated sparse algebra join attribute."""


@LATTICE_DIALECT.attr(
    'BinaryOp',
    'binary_op',
    parameters=(attr_param('value', 'string'),),
    values=('add', 'sub', 'mul', 'maximum', 'minimum'),
    summary='Coordinate-aligned sparse binary operation',
)
class BinaryOpAttr:
    """Annotated sparse binary operation attribute."""


@LATTICE_DIALECT.attr(
    'PoolMode',
    'pool_mode',
    parameters=(attr_param('value', 'string'),),
    values=('sum', 'max', 'avg'),
    summary='Sparse pooling reduction mode',
)
class PoolModeAttr:
    """Annotated sparse pooling reduction mode attribute."""


@LATTICE_DIALECT.attr(
    'VoxelReduction',
    'voxel_reduction',
    parameters=(attr_param('value', 'string'),),
    values=('sum', 'mean'),
    summary='Point-to-voxel feature aggregation reduction',
)
class VoxelReductionAttr:
    """Annotated point-to-voxel feature aggregation reduction."""


@LATTICE_DIALECT.attr(
    'PointInterpolation',
    'point_interpolation',
    parameters=(attr_param('value', 'string'),),
    values=('nearest', 'linear'),
    summary='Voxel-to-point interpolation mode',
)
class PointInterpolationAttr:
    """Annotated voxel-to-point interpolation mode."""


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


@LATTICE_DIALECT.op(
    'sparse.reindex',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('target', 'sparse_tensor'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(op_attr('fill', 'f32'),),
    assembly='functional',
    summary='Gather sparse features onto exact target support',
)
def sparse_reindex() -> None:
    """Register lattice.sparse.reindex."""


_KERNEL_ATTRS = (
    op_attr('kernel_size', 'i64_triple'),
    op_attr('stride', 'i64_triple'),
    op_attr('padding', 'i64_triple'),
    op_attr('dilation', 'i64_triple'),
)

_CONV_ATTRS = (*_KERNEL_ATTRS, op_attr('accumulation', 'str'))


@LATTICE_DIALECT.op(
    'conv3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
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
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('kernel_size', 'i64_triple'),
        op_attr('dilation', 'i64_triple'),
        op_attr('accumulation', 'str'),
    ),
    assembly='functional',
    summary='Submanifold sparse 3D convolution',
)
def subm_conv3d() -> None:
    """Register lattice.subm_conv3d."""


@LATTICE_DIALECT.op(
    'normalized_subm_conv3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('kernel_size', 'i64_triple'),
        op_attr('dilation', 'i64_triple'),
        op_attr('eps', 'f32'),
        op_attr('accumulation', 'str'),
    ),
    assembly='functional',
    summary='Weight-normalized submanifold sparse 3D convolution',
)
def normalized_subm_conv3d() -> None:
    """Register lattice.normalized_subm_conv3d."""


@LATTICE_DIALECT.op(
    'target_conv3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('target', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=_CONV_ATTRS,
    assembly='functional',
    summary='Sparse 3D convolution on explicit target coordinates',
)
def target_conv3d() -> None:
    """Register lattice.target_conv3d."""


@LATTICE_DIALECT.op(
    'conv_transpose3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=_CONV_ATTRS,
    assembly='functional',
    summary='Sparse 3D transpose convolution',
)
def conv_transpose3d() -> None:
    """Register lattice.conv_transpose3d."""


@LATTICE_DIALECT.op(
    'target_conv_transpose3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('target', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=_CONV_ATTRS,
    assembly='functional',
    summary='Sparse 3D transpose convolution on explicit target support',
)
def target_conv_transpose3d() -> None:
    """Register lattice.target_conv_transpose3d."""


@LATTICE_DIALECT.op(
    'normalized_conv_transpose3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(*_CONV_ATTRS, op_attr('eps', 'f32')),
    assembly='functional',
    summary='Weight-normalized sparse 3D transpose convolution',
)
def normalized_conv_transpose3d() -> None:
    """Register lattice.normalized_conv_transpose3d."""


@LATTICE_DIALECT.op(
    'target_normalized_conv_transpose3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('target', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(*_CONV_ATTRS, op_attr('eps', 'f32')),
    assembly='functional',
    summary='Weight-normalized transpose convolution on explicit target support',
)
def target_normalized_conv_transpose3d() -> None:
    """Register lattice.target_normalized_conv_transpose3d."""


@LATTICE_DIALECT.op(
    'generative_conv_transpose3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('kernel_size', 'i64_triple'),
        op_attr('stride', 'i64_triple'),
        op_attr('accumulation', 'str'),
    ),
    assembly='functional',
    summary='Sparse 3D generative transpose convolution',
)
def generative_conv_transpose3d() -> None:
    """Register lattice.generative_conv_transpose3d."""


@LATTICE_DIALECT.op(
    'normalized_generative_conv_transpose3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('kernel_size', 'i64_triple'),
        op_attr('stride', 'i64_triple'),
        op_attr('eps', 'f32'),
        op_attr('accumulation', 'str'),
    ),
    assembly='functional',
    summary='Weight-normalized generative sparse transpose convolution',
)
def normalized_generative_conv_transpose3d() -> None:
    """Register lattice.normalized_generative_conv_transpose3d."""


@LATTICE_DIALECT.op(
    'pool3d',
    operands=(operand('input', 'sparse_tensor'),),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('mode', 'pool_mode'),
        *_KERNEL_ATTRS,
    ),
    assembly='functional',
    summary='Local sparse 3D pooling',
)
def pool3d() -> None:
    """Register lattice.pool3d."""


@LATTICE_DIALECT.op(
    'pool_transpose3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('target', 'sparse_tensor', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=_KERNEL_ATTRS,
    assembly='functional',
    summary='Sparse average-pooling transpose',
)
def pool_transpose3d() -> None:
    """Register lattice.pool_transpose3d."""


@LATTICE_DIALECT.op(
    'trilinear_upsample3d',
    operands=(
        operand('input', 'sparse_tensor'),
        operand('target', 'sparse_tensor', optional=True),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(op_attr('stride', 'i64_triple'),),
    assembly='functional',
    summary='Normalized trilinear sparse 3D upsampling',
)
def trilinear_upsample3d() -> None:
    """Register lattice.trilinear_upsample3d."""


@LATTICE_DIALECT.op(
    'global_pool',
    operands=(operand('input', 'sparse_tensor'),),
    results=(result('result', 'tensor'),),
    attributes=(
        op_attr('mode', 'pool_mode'),
        op_attr('batch_size', 'i64'),
    ),
    assembly='functional',
    summary='Batch-wise global sparse pooling',
)
def global_pool() -> None:
    """Register lattice.global_pool."""


@LATTICE_DIALECT.op(
    'voxelize',
    operands=(
        operand('points', 'tensor'),
        operand('features', 'tensor'),
        operand('batch_indices', 'tensor'),
        operand('active_rows', 'tensor'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('voxel_size', 'f64_triple'),
        op_attr('origin', 'f64_triple'),
        op_attr('reduction', 'voxel_reduction'),
        op_attr('stride', 'i64_triple'),
    ),
    assembly='functional',
    summary='Quantize points and aggregate features into sparse voxels',
)
def voxelize() -> None:
    """Register lattice.voxelize."""


@LATTICE_DIALECT.op(
    'devoxelize',
    operands=(
        operand('points', 'tensor'),
        operand('voxels', 'sparse_tensor'),
        operand('batch_indices', 'tensor'),
        operand('point_active_rows', 'tensor'),
    ),
    results=(result('result', 'tensor'),),
    attributes=(
        op_attr('voxel_size', 'f64_triple'),
        op_attr('origin', 'f64_triple'),
        op_attr('interpolation', 'point_interpolation'),
    ),
    assembly='functional',
    summary='Interpolate sparse voxel features back to point rows',
)
def devoxelize() -> None:
    """Register lattice.devoxelize."""


@LATTICE_DIALECT.op(
    'linear',
    operands=(
        operand('input', 'tensor'),
        operand('weight', 'weight'),
        operand('bias', 'weight', optional=True),
    ),
    results=(result('result', 'tensor'),),
    assembly='functional',
    summary='Dense linear projection over feature tensors',
)
def linear() -> None:
    """Register lattice.linear."""


@LATTICE_DIALECT.op(
    'embedding_lookup',
    operands=(
        operand('input', 'tensor'),
        operand('weight', 'weight'),
    ),
    results=(result('result', 'tensor'),),
    assembly='functional',
    summary='Gather embedding rows using integer indices',
)
def embedding_lookup() -> None:
    """Register lattice.embedding_lookup."""


@LATTICE_DIALECT.op(
    'elementwise',
    operands=(operand('input', 'tensor'),),
    results=(result('result', 'tensor'),),
    attributes=(op_attr('kind', 'str'),),
    assembly='functional',
    summary='Deterministic dense elementwise transform',
)
def elementwise() -> None:
    """Register lattice.elementwise."""


@LATTICE_DIALECT.op(
    'softmax',
    operands=(operand('input', 'tensor'),),
    results=(result('result', 'tensor'),),
    attributes=(op_attr('axis', 'i64'),),
    assembly='functional',
    summary='FP32 subtract-max softmax',
)
def softmax() -> None:
    """Register lattice.softmax."""


@LATTICE_DIALECT.op(
    'activation',
    operands=(operand('input', 'tensor'),),
    results=(result('result', 'tensor'),),
    attributes=(
        op_attr('kind', 'activation'),
        op_attr('approximate', 'gelu_approx'),
        op_attr('alpha', 'f32'),
        op_attr('beta', 'f32'),
        op_attr('threshold', 'f32'),
    ),
    assembly='functional',
    summary='Dense feature activation over feature tensors',
)
def activation() -> None:
    """Register lattice.activation."""


@LATTICE_DIALECT.op(
    'batch_norm',
    operands=(
        operand('input', 'tensor'),
        operand('scale', 'weight'),
        operand('bias', 'weight'),
        operand('mean', 'weight'),
        operand('var', 'weight'),
    ),
    results=(result('result', 'tensor'),),
    attributes=(op_attr('eps', 'f32'),),
    assembly='functional',
    summary='Dense feature batch normalization with explicit frozen stats',
)
def batch_norm() -> None:
    """Register lattice.batch_norm."""


@LATTICE_DIALECT.op(
    'layer_norm',
    operands=(
        operand('input', 'tensor'),
        operand('scale', 'weight'),
        operand('bias', 'weight'),
    ),
    results=(result('result', 'tensor'),),
    attributes=(op_attr('eps', 'f32'),),
    assembly='functional',
    summary='Dense feature layer normalization',
)
def layer_norm() -> None:
    """Register lattice.layer_norm."""


@LATTICE_DIALECT.op(
    'rms_norm',
    operands=(
        operand('input', 'tensor'),
        operand('scale', 'weight'),
    ),
    results=(result('result', 'tensor'),),
    attributes=(op_attr('eps', 'f32'),),
    assembly='functional',
    summary='Dense feature RMS normalization',
)
def rms_norm() -> None:
    """Register lattice.rms_norm."""


@LATTICE_DIALECT.op(
    'sparse.binary',
    operands=(
        operand('lhs', 'sparse_tensor'),
        operand('rhs', 'sparse_tensor'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(
        op_attr('op', 'binary_op'),
        op_attr('join', 'join'),
        op_attr('lhs_fill', 'f32'),
        op_attr('rhs_fill', 'f32'),
    ),
    assembly='functional',
    summary='Coordinate-aligned sparse binary operation',
)
def sparse_binary() -> None:
    """Register lattice.sparse.binary."""


@LATTICE_DIALECT.op(
    'sparse.cat',
    operands=(
        operand('lhs', 'sparse_tensor'),
        operand('rhs', 'sparse_tensor'),
    ),
    results=(result('result', 'sparse_tensor'),),
    attributes=(op_attr('join', 'join'),),
    assembly='functional',
    summary='Coordinate-aligned sparse feature concatenation',
)
def sparse_cat() -> None:
    """Register lattice.sparse.cat."""

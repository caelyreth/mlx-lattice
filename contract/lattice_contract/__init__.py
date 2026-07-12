from __future__ import annotations

from lattice_contract.artifact import (
    ARTIFACT_GRAPH_FILE,
    ARTIFACT_WEIGHT_FILE,
    CURRENT_DIALECT_VERSION,
    DIALECT_SCHEMA_DIGEST,
)
from lattice_contract.dialect import LATTICE_DIALECT
from lattice_contract.kernel import (
    CANONICAL_CONV3D_WEIGHT_LAYOUT,
    centered_kernel_offsets,
    indexed_kernel_offsets,
    kernel_positions,
    kernel_row_permutation,
    sparse_kernel_offsets,
)
from lattice_contract.mlir import (
    MLIRModuleBuilder,
    Packing,
    SparseTensorType,
    SSAValue,
    TensorType,
    WeightType,
    dense_packing,
    quantized_packing,
)
from lattice_contract.reference import submanifold_conv3d_f32_to_f64
from lattice_contract.schema import (
    AttrDef,
    AttrParameter,
    DialectSchema,
    OpAttributeDef,
    OpDef,
    OperandDef,
    ResultDef,
    TypeDef,
    TypeParameter,
    attr_param,
    op_attr,
    operand,
    result,
    schema_digest,
    schema_fingerprint,
    type_param,
)

__version__ = '0.5.0'

__all__ = [
    'ARTIFACT_GRAPH_FILE',
    'ARTIFACT_WEIGHT_FILE',
    'CANONICAL_CONV3D_WEIGHT_LAYOUT',
    'CURRENT_DIALECT_VERSION',
    'DIALECT_SCHEMA_DIGEST',
    'LATTICE_DIALECT',
    'AttrDef',
    'AttrParameter',
    'DialectSchema',
    'MLIRModuleBuilder',
    'OpAttributeDef',
    'OpDef',
    'OperandDef',
    'Packing',
    'ResultDef',
    'SSAValue',
    'SparseTensorType',
    'TensorType',
    'TypeDef',
    'TypeParameter',
    'WeightType',
    '__version__',
    'attr_param',
    'centered_kernel_offsets',
    'dense_packing',
    'indexed_kernel_offsets',
    'kernel_positions',
    'kernel_row_permutation',
    'op_attr',
    'operand',
    'quantized_packing',
    'result',
    'schema_digest',
    'schema_fingerprint',
    'sparse_kernel_offsets',
    'submanifold_conv3d_f32_to_f64',
    'type_param',
]

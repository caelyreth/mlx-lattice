from __future__ import annotations

from lattice_contract.dialect import LATTICE_DIALECT
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
    type_param,
)

__version__ = '0.2.2'
CURRENT_DIALECT_VERSION = 0

__all__ = [
    'CURRENT_DIALECT_VERSION',
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
    'dense_packing',
    'op_attr',
    'operand',
    'quantized_packing',
    'result',
    'schema_digest',
    'type_param',
]

#ifndef LATTICE_DIALECT_LATTICE_IR_LATTICETYPES_TD
#define LATTICE_DIALECT_LATTICE_IR_LATTICETYPES_TD

include "Lattice/Dialect/Lattice/IR/LatticeDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Lattice_Type<string name, string mnemonicName>
    : TypeDef<Lattice_Dialect, name> {
  let mnemonic = mnemonicName;
  let hasCustomAssemblyFormat = 1;
}

def Lattice_SparseTensorType
    : Lattice_Type<"SparseTensor", "sparse_tensor"> {
  let summary = "Portable sparse lattice value";
  let description = [{
    A sparse tensor is a semantic value backed by ABI components:
    coordinates, features, active row count, and static sparse stride. The type
    records portable sparse conventions and excludes runtime cache identity.
  }];
  let parameters = (ins
    "unsigned":$rank,
    StringRefParameter<"coordinate convention">:$coord,
    StringRefParameter<"feature layout">:$feature,
    "mlir::Type":$dtype
  );
}

def Lattice_WeightType : Lattice_Type<"Weight", "weight"> {
  let summary = "Symbolic model weight";
  let description = [{
    A weight type records the logical consumer family and logical dtype. Tensor
    payloads live in weights.safetensors and are referenced through
    lattice.weight operations.
  }];
  let parameters = (ins
    StringRefParameter<"weight family">:$family,
    "mlir::Type":$dtype
  );
}

#endif // LATTICE_DIALECT_LATTICE_IR_LATTICETYPES_TD

#ifndef LATTICE_DIALECT_LATTICE_IR_LATTICEOPS_TD
#define LATTICE_DIALECT_LATTICE_IR_LATTICEOPS_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "Lattice/Dialect/Lattice/IR/LatticeAttrs.td"
include "Lattice/Dialect/Lattice/IR/LatticeTypes.td"

class Lattice_Op<string mnemonic, list<Trait> traits = []>
    : Op<Lattice_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

def Lattice_WeightOp : Lattice_Op<"weight", [Pure]> {
  let summary = "Bind a symbolic weight to external tensor storage";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$storage_key,
    Lattice_WeightLayoutAttr:$layout,
    Lattice_PackingAttr:$packing
  );
  let results = (outs Lattice_WeightType:$result);
  let assemblyFormat = "$sym_name attr-dict `:` type($result)";
}

def Lattice_SparseMakeOp : Lattice_Op<"sparse.make", [Pure]> {
  let summary = "Construct a sparse value from ABI tensor components";
  let arguments = (ins
    TensorOf<[I32]>:$coords,
    AnyRankedTensor:$features,
    TensorOf<[I32]>:$active,
    DenseI64ArrayAttr:$stride,
    Lattice_CoordAttr:$coord_order
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $coords `,` $features `,` $active attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_SparseDecomposeOp : Lattice_Op<"sparse.decompose", [Pure]> {
  let summary = "Expose sparse value ABI tensor components";
  let arguments = (ins Lattice_SparseTensorType:$input);
  let results = (outs
    TensorOf<[I32]>:$coords,
    AnyRankedTensor:$features,
    TensorOf<[I32]>:$active
  );
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` `(`
    type($coords) `,` type($features) `,` type($active) `)`
  }];
}

def Lattice_SparseWithFeaturesOp
    : Lattice_Op<"sparse.with_features", [Pure]> {
  let summary = "Replace sparse features while preserving coordinate support";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    AnyRankedTensor:$features
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $features attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_Conv3DOp : Lattice_Op<"conv3d"> {
  let summary = "Forward sparse 3D convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_SubmConv3DOp : Lattice_Op<"subm_conv3d"> {
  let summary = "Submanifold sparse 3D convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$dilation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_TargetConv3DOp : Lattice_Op<"target_conv3d"> {
  let summary = "Sparse 3D convolution on explicit target coordinates";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_SparseTensorType:$target,
    Lattice_WeightType:$weight,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $target `,` $weight attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_LinearOp : Lattice_Op<"linear", [Pure]> {
  let summary = "Dense linear projection over feature tensors";
  let arguments = (ins
    AnyRankedTensor:$input,
    Lattice_WeightType:$weight
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $weight attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_SparseAddOp : Lattice_Op<"sparse.add"> {
  let summary = "Coordinate-aligned sparse addition";
  let arguments = (ins
    Lattice_SparseTensorType:$lhs,
    Lattice_SparseTensorType:$rhs,
    Lattice_JoinAttr:$join,
    F32Attr:$lhs_fill,
    F32Attr:$rhs_fill
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    functional-type(operands, results)
  }];
}

#endif // LATTICE_DIALECT_LATTICE_IR_LATTICEOPS_TD

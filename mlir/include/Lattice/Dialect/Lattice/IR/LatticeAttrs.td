#ifndef LATTICE_DIALECT_LATTICE_IR_LATTICEATTRS_TD
#define LATTICE_DIALECT_LATTICE_IR_LATTICEATTRS_TD

include "Lattice/Dialect/Lattice/IR/LatticeDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Lattice_Attr<string name, string mnemonicName>
    : AttrDef<Lattice_Dialect, name> {
  let mnemonic = mnemonicName;
  let hasCustomAssemblyFormat = 1;
}

def Lattice_CoordAttr : Lattice_Attr<"Coord", "coord"> {
  let summary = "Sparse coordinate convention";
  let description = [{
    The first dialect slice supports only batch_x_y_z coordinates, matching
    sparse coordinate tensors with shape (N, 4).
  }];
  let parameters = (ins StringRefParameter<"coordinate convention">:$value);
}

def Lattice_FeatureLayoutAttr
    : Lattice_Attr<"FeatureLayout", "feature_layout"> {
  let summary = "Sparse feature matrix layout";
  let parameters = (ins StringRefParameter<"feature layout">:$value);
}

def Lattice_WeightLayoutAttr
    : Lattice_Attr<"WeightLayout", "weight_layout"> {
  let summary = "Logical weight layout";
  let parameters = (ins StringRefParameter<"weight layout">:$value);
}

def Lattice_PackingAttr : Lattice_Attr<"Packing", "packing"> {
  let summary = "Weight storage packing";
  let description = [{
    Dense weights use kind=dense. Quantized weights use kind=int4 or int8 and
    carry group size, scale dtype, and affine mode metadata.
  }];
  let parameters = (ins
    StringRefParameter<"packing kind">:$kind,
    "unsigned":$groupSize,
    "mlir::Type":$scaleType,
    StringRefParameter<"quantization mode">:$mode
  );
}

def Lattice_ActivationAttr : Lattice_Attr<"Activation", "activation"> {
  let summary = "Dense feature activation function";
  let parameters = (ins StringRefParameter<"activation function">:$value);
}

def Lattice_GeluApproxAttr : Lattice_Attr<"GeluApprox", "gelu_approx"> {
  let summary = "GELU approximation mode";
  let parameters = (ins StringRefParameter<"GELU approximation mode">:$value);
}

def Lattice_JoinAttr : Lattice_Attr<"Join", "join"> {
  let summary = "Coordinate-aligned sparse algebra join mode";
  let parameters = (ins StringRefParameter<"join mode">:$value);
}

def Lattice_BinaryOpAttr : Lattice_Attr<"BinaryOp", "binary_op"> {
  let summary = "Coordinate-aligned sparse binary operation";
  let parameters = (ins StringRefParameter<"binary operation">:$value);
}

def Lattice_PoolModeAttr : Lattice_Attr<"PoolMode", "pool_mode"> {
  let summary = "Sparse pooling reduction mode";
  let parameters = (ins StringRefParameter<"pooling mode">:$value);
}

def Lattice_VoxelReductionAttr
    : Lattice_Attr<"VoxelReduction", "voxel_reduction"> {
  let summary = "Point-to-voxel feature aggregation reduction";
  let parameters = (ins StringRefParameter<"voxel reduction mode">:$value);
}

def Lattice_PointInterpolationAttr
    : Lattice_Attr<"PointInterpolation", "point_interpolation"> {
  let summary = "Voxel-to-point interpolation mode";
  let parameters = (ins StringRefParameter<"point interpolation mode">:$value);
}

#endif // LATTICE_DIALECT_LATTICE_IR_LATTICEATTRS_TD

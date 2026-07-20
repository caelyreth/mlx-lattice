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

def Lattice_SparseReindexOp : Lattice_Op<"sparse.reindex", [Pure]> {
  let summary = "Gather sparse features onto exact target support";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_SparseTensorType:$target,
    F32Attr:$fill
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $target attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_Conv3DOp : Lattice_Op<"conv3d"> {
  let summary = "Forward sparse 3D convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_SubmConv3DOp : Lattice_Op<"subm_conv3d"> {
  let summary = "Submanifold sparse 3D convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$dilation,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_NormalizedSubmConv3DOp
    : Lattice_Op<"normalized_subm_conv3d"> {
  let summary = "Weight-normalized submanifold sparse 3D convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$dilation,
    F32Attr:$eps,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_TargetConv3DOp : Lattice_Op<"target_conv3d"> {
  let summary = "Sparse 3D convolution on explicit target coordinates";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_SparseTensorType:$target,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $target `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_ConvTranspose3DOp : Lattice_Op<"conv_transpose3d"> {
  let summary = "Sparse 3D transpose convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_TargetConvTranspose3DOp
    : Lattice_Op<"target_conv_transpose3d"> {
  let summary = "Sparse 3D transpose convolution on explicit target support";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_SparseTensorType:$target,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $target `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_NormalizedConvTranspose3DOp
    : Lattice_Op<"normalized_conv_transpose3d"> {
  let summary = "Weight-normalized sparse 3D transpose convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    F32Attr:$eps,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_TargetNormalizedConvTranspose3DOp
    : Lattice_Op<"target_normalized_conv_transpose3d"> {
  let summary = "Weight-normalized transpose convolution on target support";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_SparseTensorType:$target,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    F32Attr:$eps,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $target `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_GenerativeConvTranspose3DOp
    : Lattice_Op<"generative_conv_transpose3d"> {
  let summary = "Sparse 3D generative transpose convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_NormalizedGenerativeConvTranspose3DOp
    : Lattice_Op<"normalized_generative_conv_transpose3d"> {
  let summary = "Weight-normalized generative sparse transpose convolution";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    F32Attr:$eps,
    StrAttr:$accumulation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_Pool3DOp : Lattice_Op<"pool3d"> {
  let summary = "Local sparse 3D pooling";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_PoolModeAttr:$mode,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_PoolTranspose3DOp : Lattice_Op<"pool_transpose3d"> {
  let summary = "Sparse average-pooling transpose";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Optional<Lattice_SparseTensorType>:$target,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input (`,` $target^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_TrilinearUpsample3DOp : Lattice_Op<"trilinear_upsample3d"> {
  let summary = "Normalized trilinear sparse 3D upsampling";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Optional<Lattice_SparseTensorType>:$target,
    DenseI64ArrayAttr:$stride
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $input (`,` $target^)? attr-dict `:` functional-type(operands, results)
  }];
}

def Lattice_GlobalPoolOp : Lattice_Op<"global_pool", [Pure]> {
  let summary = "Batch-wise global sparse pooling";
  let arguments = (ins
    Lattice_SparseTensorType:$input,
    Lattice_PoolModeAttr:$mode,
    I64Attr:$batch_size
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_VoxelizeOp : Lattice_Op<"voxelize"> {
  let summary = "Quantize points and aggregate features into sparse voxels";
  let arguments = (ins
    AnyRankedTensor:$points,
    AnyRankedTensor:$features,
    TensorOf<[I32]>:$batch_indices,
    TensorOf<[I32]>:$active_rows,
    DenseF64ArrayAttr:$voxel_size,
    DenseF64ArrayAttr:$origin,
    Lattice_VoxelReductionAttr:$reduction,
    DenseI64ArrayAttr:$stride
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $points `,` $features `,` $batch_indices `,` $active_rows
    attr-dict `:` functional-type(operands, results)
  }];
}

def Lattice_DevoxelizeOp : Lattice_Op<"devoxelize"> {
  let summary = "Interpolate sparse voxel features back to point rows";
  let arguments = (ins
    AnyRankedTensor:$points,
    Lattice_SparseTensorType:$voxels,
    TensorOf<[I32]>:$batch_indices,
    TensorOf<[I32]>:$point_active_rows,
    DenseF64ArrayAttr:$voxel_size,
    DenseF64ArrayAttr:$origin,
    Lattice_PointInterpolationAttr:$interpolation
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $points `,` $voxels `,` $batch_indices `,` $point_active_rows
    attr-dict `:` functional-type(operands, results)
  }];
}

def Lattice_LinearOp : Lattice_Op<"linear", [Pure]> {
  let summary = "Dense linear projection over feature tensors";
  let arguments = (ins
    AnyRankedTensor:$input,
    Lattice_WeightType:$weight,
    Optional<Lattice_WeightType>:$bias
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_EmbeddingLookupOp : Lattice_Op<"embedding_lookup", [Pure]> {
  let summary = "Gather embedding rows using integer indices";
  let arguments = (ins
    AnyRankedTensor:$input,
    Lattice_WeightType:$weight
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $weight attr-dict `:` functional-type(operands, results)
  }];
}

def Lattice_ElementwiseOp : Lattice_Op<"elementwise", [Pure]> {
  let summary = "Deterministic dense elementwise transform";
  let arguments = (ins
    AnyRankedTensor:$input,
    StrAttr:$kind
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Lattice_SoftmaxOp : Lattice_Op<"softmax", [Pure]> {
  let summary = "FP32 subtract-max softmax";
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Lattice_ActivationOp : Lattice_Op<"activation", [Pure]> {
  let summary = "Dense feature activation over feature tensors";
  let arguments = (ins
    AnyRankedTensor:$input,
    Lattice_ActivationAttr:$kind,
    Lattice_GeluApproxAttr:$approximate,
    F32Attr:$alpha,
    F32Attr:$beta,
    F32Attr:$threshold
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_BatchNormOp : Lattice_Op<"batch_norm", [Pure]> {
  let summary = "Dense feature batch normalization with explicit frozen stats";
  let arguments = (ins
    AnyRankedTensor:$input,
    Lattice_WeightType:$scale,
    Lattice_WeightType:$bias,
    Lattice_WeightType:$mean,
    Lattice_WeightType:$var,
    F32Attr:$eps
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $scale `,` $bias `,` $mean `,` $var attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_LayerNormOp : Lattice_Op<"layer_norm", [Pure]> {
  let summary = "Dense feature layer normalization";
  let arguments = (ins
    AnyRankedTensor:$input,
    Lattice_WeightType:$scale,
    Lattice_WeightType:$bias,
    F32Attr:$eps
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $scale `,` $bias attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_RMSNormOp : Lattice_Op<"rms_norm", [Pure]> {
  let summary = "Dense feature RMS normalization";
  let arguments = (ins
    AnyRankedTensor:$input,
    Lattice_WeightType:$scale,
    F32Attr:$eps
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $scale attr-dict `:`
    functional-type(operands, results)
  }];
}

def Lattice_SparseBinaryOp : Lattice_Op<"sparse.binary"> {
  let summary = "Coordinate-aligned sparse binary operation";
  let arguments = (ins
    Lattice_SparseTensorType:$lhs,
    Lattice_SparseTensorType:$rhs,
    Lattice_BinaryOpAttr:$op,
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

def Lattice_SparseCatOp : Lattice_Op<"sparse.cat"> {
  let summary = "Coordinate-aligned sparse feature concatenation";
  let arguments = (ins
    Lattice_SparseTensorType:$lhs,
    Lattice_SparseTensorType:$rhs,
    Lattice_JoinAttr:$join
  );
  let results = (outs Lattice_SparseTensorType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    functional-type(operands, results)
  }];
}

#endif // LATTICE_DIALECT_LATTICE_IR_LATTICEOPS_TD

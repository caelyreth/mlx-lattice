#include "Lattice/Dialect/Lattice/IR/LatticeOps.h"

#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace lattice;

namespace {

LogicalResult verifyTriple(
    Operation* op,
    ArrayRef<int64_t> value,
    StringRef name,
    bool strictlyPositive
) {
    if (value.size() != 3) {
        return op->emitOpError() << name << " must contain exactly 3 integers";
    }
    for (int64_t item : value) {
        if (strictlyPositive && item <= 0) {
            return op->emitOpError()
                   << name << " values must be strictly positive";
        }
        if (!strictlyPositive && item < 0) {
            return op->emitOpError() << name << " values must be non-negative";
        }
    }
    return success();
}

LogicalResult verifySparseRank(Operation* op, SparseTensorType type) {
    if (type.getRank() != 3) {
        return op->emitOpError()
               << "only rank=3 sparse tensors are supported in v0";
    }
    if (type.getCoord() != "batch_x_y_z") {
        return op->emitOpError()
               << "only batch_x_y_z coordinate convention is supported in v0";
    }
    if (type.getFeature() != "row_channel") {
        return op->emitOpError()
               << "only row_channel feature layout is supported in v0";
    }
    return success();
}

LogicalResult
verifyWeightFamily(Operation* op, WeightType type, StringRef expected) {
    if (type.getFamily() != expected) {
        return op->emitOpError() << "expected " << expected << " weight, got "
                                 << type.getFamily();
    }
    return success();
}

LogicalResult verifyOptionalBias(Operation* op, Value bias) {
    if (!bias) {
        return success();
    }
    return verifyWeightFamily(op, cast<WeightType>(bias.getType()), "bias");
}

LogicalResult verifyChannelWeight(Operation* op, WeightType type) {
    return verifyWeightFamily(op, type, "channel");
}

LogicalResult verifyPositiveEps(Operation* op, FloatAttr eps) {
    if (eps.getValueAsDouble() <= 0.0) {
        return op->emitOpError("eps must be positive");
    }
    return success();
}

LogicalResult verifyBinaryOp(Operation* op, BinaryOpAttr binaryOp) {
    auto value = binaryOp.getValue();
    if (value != "add" && value != "sub" && value != "mul" &&
        value != "maximum" && value != "minimum") {
        return op->emitOpError()
               << "binary op must be add, sub, mul, maximum, or minimum";
    }
    return success();
}

LogicalResult verifyPoolMode(Operation* op, PoolModeAttr mode) {
    auto value = mode.getValue();
    if (value != "sum" && value != "max" && value != "avg") {
        return op->emitOpError() << "pool mode must be sum, max, or avg";
    }
    return success();
}

LogicalResult verifyVoxelReduction(Operation* op, VoxelReductionAttr mode) {
    auto value = mode.getValue();
    if (value != "sum" && value != "mean") {
        return op->emitOpError() << "voxel reduction must be sum or mean";
    }
    return success();
}

LogicalResult
verifyPointInterpolation(Operation* op, PointInterpolationAttr mode) {
    auto value = mode.getValue();
    if (value != "nearest" && value != "linear") {
        return op->emitOpError()
               << "point interpolation must be nearest or linear";
    }
    return success();
}

LogicalResult verifyActivation(Operation* op, ActivationAttr kind) {
    auto value = kind.getValue();
    if (value != "relu" && value != "sigmoid" && value != "gelu" &&
        value != "silu" && value != "leaky_relu" && value != "tanh" &&
        value != "softplus") {
        return op->emitOpError()
               << "activation must be relu, sigmoid, gelu, silu, "
                  "leaky_relu, tanh, or softplus";
    }
    return success();
}

LogicalResult verifyGeluApprox(Operation* op, GeluApproxAttr approximate) {
    auto value = approximate.getValue();
    if (value != "none" && value != "precise" && value != "tanh" &&
        value != "fast") {
        return op->emitOpError()
               << "GELU approximation must be none, precise, tanh, or fast";
    }
    return success();
}

LogicalResult verifyF64Triple(
    Operation* op,
    ArrayRef<double> value,
    StringRef name,
    bool strictlyPositive
) {
    if (value.size() != 3) {
        return op->emitOpError() << name << " must contain exactly 3 values";
    }
    for (double item : value) {
        if (strictlyPositive && item <= 0.0) {
            return op->emitOpError()
                   << name << " values must be strictly positive";
        }
    }
    return success();
}

LogicalResult verifyRank2F32(
    Operation* op,
    RankedTensorType type,
    StringRef name,
    int64_t trailingDim = ShapedType::kDynamic
) {
    if (type.getRank() != 2) {
        return op->emitOpError() << name << " must have rank 2";
    }
    if (trailingDim != ShapedType::kDynamic &&
        type.getDimSize(1) != trailingDim) {
        return op->emitOpError()
               << name << " must have trailing dimension " << trailingDim;
    }
    if (!type.getElementType().isF32()) {
        return op->emitOpError() << name << " must use f32 elements";
    }
    return success();
}

LogicalResult
verifyI32Vector(Operation* op, RankedTensorType type, StringRef name) {
    if (type.getRank() != 1) {
        return op->emitOpError() << name << " must have rank 1";
    }
    if (!type.getElementType().isInteger(32)) {
        return op->emitOpError() << name << " must use i32 elements";
    }
    return success();
}

LogicalResult
verifyI32ActiveRows(Operation* op, RankedTensorType type, StringRef name) {
    if (type.getRank() != 1 || type.getDimSize(0) != 1) {
        return op->emitOpError() << name << " must have shape (1)";
    }
    if (!type.getElementType().isInteger(32)) {
        return op->emitOpError() << name << " must use i32 elements";
    }
    return success();
}

LogicalResult
verifyRank2Tensor(Operation* op, RankedTensorType type, StringRef name) {
    if (type.getRank() != 2) {
        return op->emitOpError() << name << " must have rank 2";
    }
    return success();
}

LogicalResult verifyFeatureTensorPair(
    Operation* op,
    RankedTensorType inputType,
    RankedTensorType resultType
) {
    if (failed(verifyRank2Tensor(op, inputType, "input"))) {
        return failure();
    }
    if (failed(verifyRank2Tensor(op, resultType, "result"))) {
        return failure();
    }
    if (inputType.getElementType() != resultType.getElementType()) {
        return op->emitOpError("result dtype must match input dtype");
    }
    return success();
}

struct ConvTriples {
    ArrayRef<int64_t> kernelSize;
    ArrayRef<int64_t> stride;
    ArrayRef<int64_t> padding;
    ArrayRef<int64_t> dilation;
};

LogicalResult verifyConvTriples(Operation* op, ConvTriples triples) {
    if (failed(verifyTriple(
            op,
            triples.kernelSize,
            "kernel_size",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    if (failed(verifyTriple(
            op,
            triples.stride,
            "stride",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    if (failed(verifyTriple(
            op,
            triples.padding,
            "padding",
            /*strictlyPositive=*/false
        ))) {
        return failure();
    }
    return verifyTriple(
        op,
        triples.dilation,
        "dilation",
        /*strictlyPositive=*/true
    );
}

LogicalResult verifyTargetConv(
    Operation* op,
    SparseTensorType inputType,
    SparseTensorType targetType,
    SparseTensorType resultType,
    WeightType weightType,
    Value bias,
    ConvTriples triples,
    FloatAttr eps = {}
) {
    if (failed(verifySparseRank(op, inputType)) ||
        failed(verifySparseRank(op, targetType)) ||
        failed(verifySparseRank(op, resultType))) {
        return failure();
    }
    if (inputType.getCoord() != targetType.getCoord() ||
        targetType.getCoord() != resultType.getCoord()) {
        return op->emitOpError(
            "input, target, and result coord conventions must match"
        );
    }
    if (failed(verifyWeightFamily(op, weightType, "conv3d")) ||
        failed(verifyOptionalBias(op, bias)) ||
        failed(verifyConvTriples(op, triples))) {
        return failure();
    }
    return eps ? verifyPositiveEps(op, eps) : success();
}

} // namespace

LogicalResult WeightOp::verify() {
    auto type = cast<WeightType>(getResult().getType());
    auto layout = getLayout().getValue();
    auto packing = getPacking();

    if (getStorageKey().empty()) {
        return emitOpError("requires a non-empty storage_key");
    }
    if (type.getFamily() == "conv3d" && layout != "conv3d_o_xyz_i") {
        return emitOpError("conv3d weight must use conv3d_o_xyz_i layout");
    }
    if (type.getFamily() == "linear" && layout != "linear_o_i") {
        return emitOpError("linear weight must use linear_o_i layout");
    }
    if (type.getFamily() == "channel" && layout != "channel_c") {
        return emitOpError("channel weight must use channel_c layout");
    }
    if (type.getFamily() == "bias" && layout != "bias_c") {
        return emitOpError("bias weight must use bias_c layout");
    }
    if (type.getFamily() != "conv3d" && type.getFamily() != "linear" &&
        type.getFamily() != "channel" && type.getFamily() != "bias") {
        return emitOpError(
            "weight family must be conv3d, linear, channel, or bias"
        );
    }
    if (packing.getKind() == "dense") {
        return success();
    }
    if (type.getFamily() == "bias" || type.getFamily() == "channel") {
        return emitOpError("channel and bias weights must use dense packing");
    }
    if (packing.getKind() != "int4" && packing.getKind() != "int8") {
        return emitOpError("packing kind must be dense, int4, or int8");
    }
    if (packing.getGroupSize() == 0) {
        return emitOpError("quantized packing requires positive group_size");
    }
    if (!packing.getScaleType().isF16() && !packing.getScaleType().isF32()) {
        return emitOpError("quantized packing scale_dtype must be f16 or f32");
    }
    if (packing.getMode() != "affine") {
        return emitOpError("only affine quantized packing is supported in v0");
    }
    return success();
}

LogicalResult SparseMakeOp::verify() {
    auto coordsType = cast<RankedTensorType>(getCoords().getType());
    auto featuresType = cast<RankedTensorType>(getFeatures().getType());
    auto activeType = cast<RankedTensorType>(getActive().getType());
    auto resultType = cast<SparseTensorType>(getResult().getType());

    if (coordsType.getRank() != 2 || coordsType.getDimSize(1) != 4) {
        return emitOpError("coords must have shape (?, 4)");
    }
    if (featuresType.getRank() != 2) {
        return emitOpError("features must have rank 2");
    }
    if (coordsType.hasStaticShape() && featuresType.hasStaticShape() &&
        coordsType.getDimSize(0) != featuresType.getDimSize(0)) {
        return emitOpError("coords/features capacities must match");
    }
    if (activeType.getRank() != 1 || activeType.getDimSize(0) != 1) {
        return emitOpError("active must have shape (1)");
    }
    if (failed(verifyTriple(
            getOperation(),
            getStride(),
            "stride",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    if (getCoordOrder().getValue() != resultType.getCoord()) {
        return emitOpError("coord_order must match sparse result type");
    }
    if (featuresType.getElementType() != resultType.getDtype()) {
        return emitOpError("feature dtype must match sparse result dtype");
    }
    return verifySparseRank(getOperation(), resultType);
}

LogicalResult SparseDecomposeOp::verify() {
    return verifySparseRank(getOperation(), getInput().getType());
}

LogicalResult SparseWithFeaturesOp::verify() {
    auto inputType = getInput().getType();
    auto resultType = getResult().getType();
    auto featuresType = cast<RankedTensorType>(getFeatures().getType());

    if (failed(verifySparseRank(getOperation(), inputType))) {
        return failure();
    }
    if (inputType.getRank() != resultType.getRank() ||
        inputType.getCoord() != resultType.getCoord() ||
        inputType.getFeature() != resultType.getFeature()) {
        return emitOpError("result sparse support must match input support");
    }
    if (featuresType.getRank() != 2) {
        return emitOpError("replacement features must have rank 2");
    }
    if (featuresType.getElementType() != resultType.getDtype()) {
        return emitOpError("replacement feature dtype must match result dtype");
    }
    return success();
}

LogicalResult SparseReindexOp::verify() {
    auto inputType = getInput().getType();
    auto targetType = getTarget().getType();
    auto resultType = getResult().getType();

    if (failed(verifySparseRank(getOperation(), inputType)) ||
        failed(verifySparseRank(getOperation(), targetType)) ||
        failed(verifySparseRank(getOperation(), resultType))) {
        return failure();
    }
    if (inputType.getCoord() != targetType.getCoord() ||
        inputType.getFeature() != targetType.getFeature()) {
        return emitOpError("input and target must share sparse conventions");
    }
    if (inputType.getDtype() != resultType.getDtype()) {
        return emitOpError("result dtype must match input feature dtype");
    }
    return success();
}

LogicalResult Conv3DOp::verify() {
    if (failed(verifySparseRank(getOperation(), getInput().getType()))) {
        return failure();
    }
    if (failed(verifySparseRank(getOperation(), getResult().getType()))) {
        return failure();
    }
    if (failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "conv3d")
        )) {
        return failure();
    }
    if (failed(verifyOptionalBias(getOperation(), getBias()))) {
        return failure();
    }
    return verifyConvTriples(
        getOperation(),
        ConvTriples{
            .kernelSize = getKernelSize(),
            .stride = getStride(),
            .padding = getPadding(),
            .dilation = getDilation(),
        }
    );
}

LogicalResult SubmConv3DOp::verify() {
    auto kernelSize = getKernelSize();
    if (failed(verifySparseRank(getOperation(), getInput().getType()))) {
        return failure();
    }
    if (failed(verifySparseRank(getOperation(), getResult().getType()))) {
        return failure();
    }
    if (failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "conv3d")
        )) {
        return failure();
    }
    if (failed(verifyOptionalBias(getOperation(), getBias()))) {
        return failure();
    }
    if (getOperation()->hasAttr("stride") ||
        getOperation()->hasAttr("padding")) {
        return emitOpError("must not carry stride or padding");
    }
    if (failed(verifyTriple(
            getOperation(),
            kernelSize,
            "kernel_size",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    for (int64_t item : kernelSize) {
        if ((item % 2) == 0) {
            return emitOpError("submanifold kernel_size values must be odd");
        }
    }
    return verifyTriple(
        getOperation(),
        getDilation(),
        "dilation",
        /*strictlyPositive=*/true
    );
}

LogicalResult NormalizedSubmConv3DOp::verify() {
    auto kernelSize = getKernelSize();
    if (failed(verifySparseRank(getOperation(), getInput().getType())) ||
        failed(verifySparseRank(getOperation(), getResult().getType())) ||
        failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "conv3d")
        ) ||
        failed(verifyOptionalBias(getOperation(), getBias()))) {
        return failure();
    }
    if (getOperation()->hasAttr("stride") ||
        getOperation()->hasAttr("padding")) {
        return emitOpError("must not carry stride or padding");
    }
    if (failed(verifyTriple(
            getOperation(),
            kernelSize,
            "kernel_size",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    for (int64_t item : kernelSize) {
        if ((item % 2) == 0) {
            return emitOpError("submanifold kernel_size values must be odd");
        }
    }
    if (failed(verifyTriple(
            getOperation(),
            getDilation(),
            "dilation",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    return verifyPositiveEps(getOperation(), getEpsAttr());
}

LogicalResult TargetConv3DOp::verify() {
    return verifyTargetConv(
        getOperation(),
        getInput().getType(),
        getTarget().getType(),
        getResult().getType(),
        getWeight().getType(),
        getBias(),
        ConvTriples{
            .kernelSize = getKernelSize(),
            .stride = getStride(),
            .padding = getPadding(),
            .dilation = getDilation(),
        }
    );
}

LogicalResult TargetConvTranspose3DOp::verify() {
    return verifyTargetConv(
        getOperation(),
        getInput().getType(),
        getTarget().getType(),
        getResult().getType(),
        getWeight().getType(),
        getBias(),
        ConvTriples{
            .kernelSize = getKernelSize(),
            .stride = getStride(),
            .padding = getPadding(),
            .dilation = getDilation(),
        }
    );
}

LogicalResult ConvTranspose3DOp::verify() {
    if (failed(verifySparseRank(getOperation(), getInput().getType()))) {
        return failure();
    }
    if (failed(verifySparseRank(getOperation(), getResult().getType()))) {
        return failure();
    }
    if (failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "conv3d")
        )) {
        return failure();
    }
    if (failed(verifyOptionalBias(getOperation(), getBias()))) {
        return failure();
    }
    return verifyConvTriples(
        getOperation(),
        ConvTriples{
            .kernelSize = getKernelSize(),
            .stride = getStride(),
            .padding = getPadding(),
            .dilation = getDilation(),
        }
    );
}

LogicalResult NormalizedConvTranspose3DOp::verify() {
    if (failed(verifySparseRank(getOperation(), getInput().getType())) ||
        failed(verifySparseRank(getOperation(), getResult().getType())) ||
        failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "conv3d")
        ) ||
        failed(verifyOptionalBias(getOperation(), getBias()))) {
        return failure();
    }
    if (failed(verifyConvTriples(
            getOperation(),
            ConvTriples{
                .kernelSize = getKernelSize(),
                .stride = getStride(),
                .padding = getPadding(),
                .dilation = getDilation(),
            }
        ))) {
        return failure();
    }
    return verifyPositiveEps(getOperation(), getEpsAttr());
}

LogicalResult TargetNormalizedConvTranspose3DOp::verify() {
    return verifyTargetConv(
        getOperation(),
        getInput().getType(),
        getTarget().getType(),
        getResult().getType(),
        getWeight().getType(),
        getBias(),
        ConvTriples{
            .kernelSize = getKernelSize(),
            .stride = getStride(),
            .padding = getPadding(),
            .dilation = getDilation(),
        },
        getEpsAttr()
    );
}

LogicalResult GenerativeConvTranspose3DOp::verify() {
    if (failed(verifySparseRank(getOperation(), getInput().getType()))) {
        return failure();
    }
    if (failed(verifySparseRank(getOperation(), getResult().getType()))) {
        return failure();
    }
    if (failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "conv3d")
        )) {
        return failure();
    }
    if (failed(verifyOptionalBias(getOperation(), getBias()))) {
        return failure();
    }
    if (getOperation()->hasAttr("padding") ||
        getOperation()->hasAttr("dilation")) {
        return emitOpError("must not carry padding or dilation");
    }
    if (failed(verifyTriple(
            getOperation(),
            getKernelSize(),
            "kernel_size",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    return verifyTriple(
        getOperation(),
        getStride(),
        "stride",
        /*strictlyPositive=*/true
    );
}

LogicalResult NormalizedGenerativeConvTranspose3DOp::verify() {
    if (failed(verifySparseRank(getOperation(), getInput().getType())) ||
        failed(verifySparseRank(getOperation(), getResult().getType())) ||
        failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "conv3d")
        ) ||
        failed(verifyOptionalBias(getOperation(), getBias()))) {
        return failure();
    }
    if (getOperation()->hasAttr("padding") ||
        getOperation()->hasAttr("dilation")) {
        return emitOpError("must not carry padding or dilation");
    }
    if (failed(verifyTriple(
            getOperation(),
            getKernelSize(),
            "kernel_size",
            /*strictlyPositive=*/true
        )) ||
        failed(verifyTriple(
            getOperation(),
            getStride(),
            "stride",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    return verifyPositiveEps(getOperation(), getEpsAttr());
}

LogicalResult Pool3DOp::verify() {
    if (failed(verifySparseRank(getOperation(), getInput().getType()))) {
        return failure();
    }
    if (failed(verifySparseRank(getOperation(), getResult().getType()))) {
        return failure();
    }
    if (failed(verifyPoolMode(getOperation(), getMode()))) {
        return failure();
    }
    return verifyConvTriples(
        getOperation(),
        ConvTriples{
            .kernelSize = getKernelSize(),
            .stride = getStride(),
            .padding = getPadding(),
            .dilation = getDilation(),
        }
    );
}

LogicalResult PoolTranspose3DOp::verify() {
    if (failed(verifySparseRank(getOperation(), getInput().getType())) ||
        failed(verifySparseRank(getOperation(), getResult().getType()))) {
        return failure();
    }
    if (getTarget() &&
        failed(verifySparseRank(getOperation(), getTarget().getType()))) {
        return failure();
    }
    return verifyConvTriples(
        getOperation(),
        ConvTriples{
            .kernelSize = getKernelSize(),
            .stride = getStride(),
            .padding = getPadding(),
            .dilation = getDilation(),
        }
    );
}

LogicalResult TrilinearUpsample3DOp::verify() {
    if (failed(verifySparseRank(getOperation(), getInput().getType())) ||
        failed(verifySparseRank(getOperation(), getResult().getType()))) {
        return failure();
    }
    if (getTarget() &&
        failed(verifySparseRank(getOperation(), getTarget().getType()))) {
        return failure();
    }
    return verifyTriple(
        getOperation(), getStride(), "stride", /*strictlyPositive=*/true
    );
}

LogicalResult GlobalPoolOp::verify() {
    auto inputType = getInput().getType();
    auto resultType = cast<RankedTensorType>(getResult().getType());

    if (failed(verifySparseRank(getOperation(), inputType))) {
        return failure();
    }
    if (failed(verifyPoolMode(getOperation(), getMode()))) {
        return failure();
    }
    auto batchSize = getBatchSizeAttr().getValue().getSExtValue();
    if (batchSize < -1) {
        return emitOpError("batch_size must be -1 or non-negative");
    }
    if (resultType.getRank() != 2) {
        return emitOpError("result must have rank 2");
    }
    if (resultType.getElementType() != inputType.getDtype()) {
        return emitOpError("result dtype must match sparse feature dtype");
    }
    return success();
}

LogicalResult VoxelizeOp::verify() {
    auto pointsType = cast<RankedTensorType>(getPoints().getType());
    auto featuresType = cast<RankedTensorType>(getFeatures().getType());
    auto batchType = cast<RankedTensorType>(getBatchIndices().getType());
    auto activeType = cast<RankedTensorType>(getActiveRows().getType());
    auto resultType = getResult().getType();

    if (failed(verifyRank2F32(
            getOperation(), pointsType, "points", /*trailingDim=*/3
        ))) {
        return failure();
    }
    if (failed(verifyRank2F32(getOperation(), featuresType, "features"))) {
        return failure();
    }
    if (!featuresType.getElementType().isF32()) {
        return emitOpError("features must use f32 elements");
    }
    if (pointsType.hasStaticShape() && featuresType.hasStaticShape() &&
        pointsType.getDimSize(0) != featuresType.getDimSize(0)) {
        return emitOpError("points/features row counts must match");
    }
    if (failed(verifyI32Vector(getOperation(), batchType, "batch_indices"))) {
        return failure();
    }
    if (failed(
            verifyI32ActiveRows(getOperation(), activeType, "active_rows")
        )) {
        return failure();
    }
    if (failed(verifyF64Triple(
            getOperation(),
            getVoxelSize(),
            "voxel_size",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    if (failed(verifyF64Triple(
            getOperation(),
            getOrigin(),
            "origin",
            /*strictlyPositive=*/false
        ))) {
        return failure();
    }
    if (failed(verifyVoxelReduction(getOperation(), getReduction()))) {
        return failure();
    }
    if (failed(verifyTriple(
            getOperation(),
            getStride(),
            "stride",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    if (resultType.getDtype() != featuresType.getElementType()) {
        return emitOpError("result dtype must match feature dtype");
    }
    return verifySparseRank(getOperation(), resultType);
}

LogicalResult DevoxelizeOp::verify() {
    auto pointsType = cast<RankedTensorType>(getPoints().getType());
    auto voxelType = getVoxels().getType();
    auto batchType = cast<RankedTensorType>(getBatchIndices().getType());
    auto activeType = cast<RankedTensorType>(getPointActiveRows().getType());
    auto resultType = cast<RankedTensorType>(getResult().getType());

    if (failed(verifyRank2F32(
            getOperation(), pointsType, "points", /*trailingDim=*/3
        ))) {
        return failure();
    }
    if (failed(verifySparseRank(getOperation(), voxelType))) {
        return failure();
    }
    if (failed(verifyI32Vector(getOperation(), batchType, "batch_indices"))) {
        return failure();
    }
    if (failed(
            verifyI32ActiveRows(getOperation(), activeType, "point_active_rows")
        )) {
        return failure();
    }
    if (failed(verifyF64Triple(
            getOperation(),
            getVoxelSize(),
            "voxel_size",
            /*strictlyPositive=*/true
        ))) {
        return failure();
    }
    if (failed(verifyF64Triple(
            getOperation(),
            getOrigin(),
            "origin",
            /*strictlyPositive=*/false
        ))) {
        return failure();
    }
    if (failed(verifyPointInterpolation(getOperation(), getInterpolation()))) {
        return failure();
    }
    if (resultType.getRank() != 2) {
        return emitOpError("result must have rank 2");
    }
    if (resultType.getElementType() != voxelType.getDtype()) {
        return emitOpError("result dtype must match voxel feature dtype");
    }
    return success();
}

LogicalResult LinearOp::verify() {
    auto inputType = cast<RankedTensorType>(getInput().getType());
    auto resultType = cast<RankedTensorType>(getResult().getType());

    if (failed(
            verifyFeatureTensorPair(getOperation(), inputType, resultType)
        )) {
        return failure();
    }
    if (failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "linear")
        )) {
        return failure();
    }
    return verifyOptionalBias(getOperation(), getBias());
}

LogicalResult ActivationOp::verify() {
    auto inputType = cast<RankedTensorType>(getInput().getType());
    auto resultType = cast<RankedTensorType>(getResult().getType());
    auto alpha =
        getOperation()->getAttrOfType<FloatAttr>("alpha").getValueAsDouble();
    auto beta =
        getOperation()->getAttrOfType<FloatAttr>("beta").getValueAsDouble();

    if (failed(
            verifyFeatureTensorPair(getOperation(), inputType, resultType)
        )) {
        return failure();
    }
    if (failed(verifyActivation(getOperation(), getKind()))) {
        return failure();
    }
    if (failed(verifyGeluApprox(getOperation(), getApproximate()))) {
        return failure();
    }
    if (alpha < 0.0) {
        return emitOpError("alpha must be non-negative");
    }
    if (beta <= 0.0) {
        return emitOpError("beta must be positive");
    }
    return success();
}

LogicalResult BatchNormOp::verify() {
    auto inputType = cast<RankedTensorType>(getInput().getType());
    auto resultType = cast<RankedTensorType>(getResult().getType());

    if (failed(
            verifyFeatureTensorPair(getOperation(), inputType, resultType)
        )) {
        return failure();
    }
    if (failed(verifyChannelWeight(getOperation(), getScale().getType())) ||
        failed(
            verifyWeightFamily(getOperation(), getBias().getType(), "bias")
        ) ||
        failed(verifyChannelWeight(getOperation(), getMean().getType())) ||
        failed(verifyChannelWeight(getOperation(), getVar().getType()))) {
        return failure();
    }
    return verifyPositiveEps(
        getOperation(), getOperation()->getAttrOfType<FloatAttr>("eps")
    );
}

LogicalResult LayerNormOp::verify() {
    auto inputType = cast<RankedTensorType>(getInput().getType());
    auto resultType = cast<RankedTensorType>(getResult().getType());

    if (failed(
            verifyFeatureTensorPair(getOperation(), inputType, resultType)
        )) {
        return failure();
    }
    if (failed(verifyChannelWeight(getOperation(), getScale().getType()))) {
        return failure();
    }
    if (failed(
            verifyWeightFamily(getOperation(), getBias().getType(), "bias")
        )) {
        return failure();
    }
    return verifyPositiveEps(
        getOperation(), getOperation()->getAttrOfType<FloatAttr>("eps")
    );
}

LogicalResult RMSNormOp::verify() {
    auto inputType = cast<RankedTensorType>(getInput().getType());
    auto resultType = cast<RankedTensorType>(getResult().getType());

    if (failed(
            verifyFeatureTensorPair(getOperation(), inputType, resultType)
        )) {
        return failure();
    }
    if (failed(verifyChannelWeight(getOperation(), getScale().getType()))) {
        return failure();
    }
    return verifyPositiveEps(
        getOperation(), getOperation()->getAttrOfType<FloatAttr>("eps")
    );
}

LogicalResult SparseBinaryOp::verify() {
    auto lhsType = getLhs().getType();
    auto rhsType = getRhs().getType();
    auto resultType = getResult().getType();
    auto join = getJoin().getValue();

    if (failed(verifySparseRank(getOperation(), lhsType)) ||
        failed(verifySparseRank(getOperation(), rhsType)) ||
        failed(verifySparseRank(getOperation(), resultType))) {
        return failure();
    }
    if (lhsType.getCoord() != rhsType.getCoord() ||
        lhsType.getFeature() != rhsType.getFeature()) {
        return emitOpError(
            "sparse binary operands must share sparse conventions"
        );
    }
    if (failed(verifyBinaryOp(getOperation(), getOp()))) {
        return failure();
    }
    if (join != "inner" && join != "left" && join != "right" &&
        join != "outer") {
        return emitOpError("join must be inner, left, right, or outer");
    }
    return success();
}

LogicalResult SparseCatOp::verify() {
    auto lhsType = getLhs().getType();
    auto rhsType = getRhs().getType();
    auto resultType = getResult().getType();
    auto join = getJoin().getValue();

    if (failed(verifySparseRank(getOperation(), lhsType)) ||
        failed(verifySparseRank(getOperation(), rhsType)) ||
        failed(verifySparseRank(getOperation(), resultType))) {
        return failure();
    }
    if (lhsType.getCoord() != rhsType.getCoord() ||
        lhsType.getFeature() != rhsType.getFeature() ||
        lhsType.getDtype() != rhsType.getDtype()) {
        return emitOpError(
            "sparse cat operands must share sparse conventions and dtype"
        );
    }
    if (resultType.getCoord() != lhsType.getCoord() ||
        resultType.getFeature() != lhsType.getFeature() ||
        resultType.getDtype() != lhsType.getDtype()) {
        return emitOpError(
            "sparse cat result must preserve sparse conventions and dtype"
        );
    }
    if (join != "inner" && join != "left" && join != "right" &&
        join != "outer") {
        return emitOpError("join must be inner, left, right, or outer");
    }
    return success();
}

#define GET_OP_CLASSES
#include "Lattice/Dialect/Lattice/IR/LatticeOps.cpp.inc"

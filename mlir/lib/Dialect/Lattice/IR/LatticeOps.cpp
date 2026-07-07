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

} // namespace

LogicalResult WeightOp::verify() {
    auto type = cast<WeightType>(getResult().getType());
    auto layout = getLayout().getValue();
    auto packing = getPacking();

    if (getStorageKey().empty()) {
        return emitOpError("requires a non-empty storage_key");
    }
    if (type.getFamily() == "conv3d" && layout != "conv3d_o_zyx_i") {
        return emitOpError("conv3d weight must use conv3d_o_zyx_i layout");
    }
    if (type.getFamily() == "linear" && layout != "linear_o_i") {
        return emitOpError("linear weight must use linear_o_i layout");
    }
    if (packing.getKind() == "dense") {
        return success();
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

LogicalResult TargetConv3DOp::verify() {
    auto inputType = getInput().getType();
    auto targetType = getTarget().getType();
    auto resultType = getResult().getType();

    if (failed(verifySparseRank(getOperation(), inputType)) ||
        failed(verifySparseRank(getOperation(), targetType)) ||
        failed(verifySparseRank(getOperation(), resultType))) {
        return failure();
    }
    if (inputType.getCoord() != targetType.getCoord() ||
        targetType.getCoord() != resultType.getCoord()) {
        return emitOpError(
            "input, target, and result coord conventions must match"
        );
    }
    if (failed(
            verifyWeightFamily(getOperation(), getWeight().getType(), "conv3d")
        )) {
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

LogicalResult LinearOp::verify() {
    return verifyWeightFamily(getOperation(), getWeight().getType(), "linear");
}

LogicalResult SparseAddOp::verify() {
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
        return emitOpError("sparse add operands must share sparse conventions");
    }
    if (join != "inner" && join != "left" && join != "right" &&
        join != "outer") {
        return emitOpError("join must be inner, left, right, or outer");
    }
    return success();
}

#define GET_OP_CLASSES
#include "Lattice/Dialect/Lattice/IR/LatticeOps.cpp.inc"

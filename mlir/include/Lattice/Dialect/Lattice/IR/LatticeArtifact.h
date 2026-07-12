#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace lattice {

inline constexpr int kCurrentArtifactIRVersion = 1;
inline constexpr llvm::StringLiteral kArtifactSchemaDigest =
    "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa";
inline constexpr llvm::StringLiteral kArtifactWeightFile =
    "weights.safetensors";

mlir::LogicalResult verifyArtifactContract(mlir::ModuleOp module);

} // namespace lattice

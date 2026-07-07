#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace lattice {

inline constexpr int kCurrentArtifactIRVersion = 0;
inline constexpr llvm::StringLiteral kArtifactSchemaDigest =
    "81c8424987d97d0f6cd514b50d8db2307e467e55f32a6c30dfdf6e311d565443";
inline constexpr llvm::StringLiteral kArtifactWeightFile =
    "weights.safetensors";

mlir::LogicalResult verifyArtifactContract(mlir::ModuleOp module);

} // namespace lattice

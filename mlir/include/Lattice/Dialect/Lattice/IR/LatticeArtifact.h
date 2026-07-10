#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace lattice {

inline constexpr int kCurrentArtifactIRVersion = 0;
inline constexpr llvm::StringLiteral kArtifactSchemaDigest =
    "33a97d62e5b150b98940c62284f42b326e879cc4aca2747cdbc0d77c851f66c7";
inline constexpr llvm::StringLiteral kArtifactWeightFile =
    "weights.safetensors";

mlir::LogicalResult verifyArtifactContract(mlir::ModuleOp module);

} // namespace lattice

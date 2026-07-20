#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace lattice {

inline constexpr int kCurrentArtifactIRVersion = 2;
inline constexpr llvm::StringLiteral kArtifactSchemaDigest =
    "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad";
inline constexpr llvm::StringLiteral kArtifactWeightFile =
    "weights.safetensors";

mlir::LogicalResult verifyArtifactContract(mlir::ModuleOp module);

} // namespace lattice

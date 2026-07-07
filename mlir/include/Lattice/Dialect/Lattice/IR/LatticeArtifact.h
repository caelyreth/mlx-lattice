#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace lattice {

inline constexpr int kCurrentArtifactIRVersion = 0;
inline constexpr llvm::StringLiteral kArtifactWeightFile =
    "weights.safetensors";

mlir::LogicalResult verifyArtifactContract(mlir::ModuleOp module);

} // namespace lattice

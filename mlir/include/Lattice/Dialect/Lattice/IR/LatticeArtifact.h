#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace lattice {

inline constexpr int kCurrentArtifactIRVersion = 0;
inline constexpr llvm::StringLiteral kArtifactSchemaDigest =
    "545fd43029e2fbebc18404babc0ec463730a13056a18f51280d8f1a272a3369c";
inline constexpr llvm::StringLiteral kArtifactWeightFile =
    "weights.safetensors";

mlir::LogicalResult verifyArtifactContract(mlir::ModuleOp module);

} // namespace lattice

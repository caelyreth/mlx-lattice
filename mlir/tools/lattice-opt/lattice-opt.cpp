#include "Lattice/Dialect/Lattice/IR/LatticeArtifact.h"
#include "Lattice/Dialect/Lattice/IR/LatticeDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace {

struct VerifyLatticeArtifactPass : mlir::PassWrapper<
                                       VerifyLatticeArtifactPass,
                                       mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyLatticeArtifactPass)

    llvm::StringRef getArgument() const final {
        return "lattice-verify-artifact";
    }

    llvm::StringRef getDescription() const final {
        return "verify the lattice artifact module-level ABI contract";
    }

    void runOnOperation() final {
        if (mlir::failed(lattice::verifyArtifactContract(getOperation()))) {
            signalPassFailure();
        }
    }
};

} // namespace

static mlir::PassRegistration<VerifyLatticeArtifactPass>
    verifyLatticeArtifactPass;

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, lattice::LatticeDialect>();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(
            argc, argv, "Lattice MLIR optimizer driver\n", registry
        )
    );
}

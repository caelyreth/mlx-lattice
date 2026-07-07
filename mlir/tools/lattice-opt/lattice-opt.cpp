#include "Lattice/Dialect/Lattice/IR/LatticeDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, lattice::LatticeDialect>();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(
            argc, argv, "Lattice MLIR optimizer driver\n", registry
        )
    );
}

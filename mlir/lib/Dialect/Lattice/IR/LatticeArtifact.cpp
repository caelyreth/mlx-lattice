#include "Lattice/Dialect/Lattice/IR/LatticeArtifact.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace lattice {

mlir::LogicalResult verifyArtifactContract(mlir::ModuleOp module) {
    auto irVersionAttr =
        module->getAttrOfType<mlir::IntegerAttr>("lattice.ir_version");
    if (!irVersionAttr) {
        return module.emitError()
               << "lattice artifact module requires lattice.ir_version = "
               << kCurrentArtifactIRVersion;
    }
    if (irVersionAttr.getInt() != kCurrentArtifactIRVersion) {
        return module.emitError()
               << "unsupported lattice.ir_version: " << irVersionAttr.getInt()
               << " (expected " << kCurrentArtifactIRVersion << ")";
    }

    auto weightFileAttr =
        module->getAttrOfType<mlir::StringAttr>("lattice.weight_file");
    if (!weightFileAttr) {
        return module.emitError()
               << "lattice artifact module requires lattice.weight_file = \""
               << kArtifactWeightFile << "\"";
    }
    if (weightFileAttr.getValue() != kArtifactWeightFile) {
        return module.emitError()
               << "unsupported lattice.weight_file: \""
               << weightFileAttr.getValue() << "\" (expected \""
               << kArtifactWeightFile << "\")";
    }

    auto functions = module.getOps<mlir::func::FuncOp>();
    auto functionIt = functions.begin();
    if (functionIt == functions.end()) {
        return module.emitError()
               << "lattice artifact module requires exactly one func.func "
                  "entry named @forward";
    }
    auto function = *functionIt;
    if (++functionIt != functions.end()) {
        return module.emitError()
               << "lattice artifact module supports exactly one func.func "
                  "entry";
    }
    if (function.getSymName() != "forward") {
        return function.emitError()
               << "lattice artifact entry function must be named @forward";
    }
    if (function.getFunctionType().getNumResults() == 0) {
        return function.emitError()
               << "lattice artifact entry function must return at least one "
                  "value";
    }
    if (function.empty()) {
        return function.emitError()
               << "lattice artifact entry function must have a body";
    }
    if (!function.getBody().hasOneBlock()) {
        return function.emitError()
               << "lattice artifact entry function must contain exactly one "
                  "block";
    }

    for (auto& operation : function.front()) {
        if (mlir::isa<mlir::func::ReturnOp>(operation)) {
            continue;
        }
        if (operation.getName().getDialectNamespace() != "lattice") {
            return operation.emitError()
                   << "lattice artifact entry function may contain only "
                      "lattice operations and func.return";
        }
    }

    return mlir::success();
}

} // namespace lattice

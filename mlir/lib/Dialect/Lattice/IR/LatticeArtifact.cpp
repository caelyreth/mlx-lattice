#include "Lattice/Dialect/Lattice/IR/LatticeArtifact.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

namespace lattice {

namespace {

bool stringArrayAttr(
    mlir::ModuleOp module,
    llvm::StringRef name,
    llvm::SmallVectorImpl<llvm::StringRef>& values
) {
    auto attr = module->getAttrOfType<mlir::ArrayAttr>(name);
    if (!attr) {
        return false;
    }
    values.reserve(attr.size());
    for (auto item : attr) {
        auto string = mlir::dyn_cast<mlir::StringAttr>(item);
        if (!string) {
            return false;
        }
        values.push_back(string.getValue());
    }
    return true;
}

mlir::LogicalResult verifyStringArrayAttr(
    mlir::ModuleOp module,
    mlir::func::FuncOp function,
    llvm::StringRef namesAttrName,
    llvm::StringRef rolesAttrName,
    unsigned expectedSize,
    llvm::ArrayRef<llvm::StringRef> allowedRoles
) {
    llvm::SmallVector<llvm::StringRef> names;
    if (!stringArrayAttr(module, namesAttrName, names)) {
        return module.emitError()
               << "lattice artifact module requires string array attribute "
               << namesAttrName;
    }
    llvm::SmallVector<llvm::StringRef> roles;
    if (!stringArrayAttr(module, rolesAttrName, roles)) {
        return module.emitError()
               << "lattice artifact module requires string array attribute "
               << rolesAttrName;
    }
    if (names.size() != expectedSize) {
        return function.emitError() << namesAttrName << " must contain "
                                    << expectedSize << " entries";
    }
    if (roles.size() != expectedSize) {
        return function.emitError() << rolesAttrName << " must contain "
                                    << expectedSize << " entries";
    }

    llvm::StringSet<> seenNames;
    llvm::StringSet<> allowed;
    for (auto role : allowedRoles) {
        allowed.insert(role);
    }
    for (auto [name, role] : llvm::zip(names, roles)) {
        if (name.empty()) {
            return module.emitError()
                   << namesAttrName << " must not contain empty names";
        }
        if (!seenNames.insert(name).second) {
            return module.emitError()
                   << namesAttrName << " contains duplicate name \"" << name
                   << "\"";
        }
        if (!allowed.contains(role)) {
            return module.emitError()
                   << rolesAttrName << " contains unsupported role \"" << role
                   << "\"";
        }
    }
    return mlir::success();
}

} // namespace

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

    auto schemaDigestAttr =
        module->getAttrOfType<mlir::StringAttr>("lattice.schema_digest");
    if (!schemaDigestAttr) {
        return module.emitError()
               << "lattice artifact module requires lattice.schema_digest = \""
               << kArtifactSchemaDigest << "\"";
    }
    if (schemaDigestAttr.getValue() != kArtifactSchemaDigest) {
        return module.emitError()
               << "unsupported lattice.schema_digest: \""
               << schemaDigestAttr.getValue() << "\" (expected \""
               << kArtifactSchemaDigest << "\")";
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

    if (mlir::failed(verifyStringArrayAttr(
            module,
            function,
            "lattice.input_names",
            "lattice.input_roles",
            function.getNumArguments(),
            {"tensor", "sparse_coords", "sparse_features", "sparse_active"}
        ))) {
        return mlir::failure();
    }
    if (mlir::failed(verifyStringArrayAttr(
            module,
            function,
            "lattice.output_names",
            "lattice.output_roles",
            function.getFunctionType().getNumResults(),
            {"tensor", "sparse_tensor"}
        ))) {
        return mlir::failure();
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

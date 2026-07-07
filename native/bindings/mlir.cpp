#include "bindings/registrations.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

#include "Lattice/Dialect/Lattice/IR/LatticeArtifact.h"
#include "Lattice/Dialect/Lattice/IR/LatticeDialect.h"
#include "Lattice/Dialect/Lattice/IR/LatticeOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlx_lattice::bindings {

namespace {

void register_lattice_dialects(mlir::DialectRegistry& registry) {
    registry.insert<mlir::func::FuncDialect, lattice::LatticeDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp>
parse_verified_module(std::string const& graph, mlir::MLIRContext& context) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(graph), &context
    );
    if (!module) {
        return {};
    }
    if (mlir::failed(module->verify())) {
        return {};
    }
    if (mlir::failed(lattice::verifyArtifactContract(*module))) {
        return {};
    }
    return module;
}

template <typename Fn>
auto with_verified_module(std::string const& graph, Fn&& fn) {
    mlir::DialectRegistry registry;
    register_lattice_dialects(registry);

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    std::string diagnostics;
    llvm::raw_string_ostream stream(diagnostics);
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic& diagnostic) {
            diagnostic.print(stream);
            stream << "\n";
        }
    );

    auto module = parse_verified_module(graph, context);
    if (!module) {
        stream.flush();
        throw nb::value_error(
            diagnostics.empty() ? "invalid lattice MLIR" : diagnostics.c_str()
        );
    }
    return fn(*module);
}

std::string parse_error_message(std::string const& graph) {
    mlir::DialectRegistry registry;
    register_lattice_dialects(registry);

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    std::string diagnostics;
    llvm::raw_string_ostream stream(diagnostics);
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic& diagnostic) {
            diagnostic.print(stream);
            stream << "\n";
        }
    );

    auto module = parse_verified_module(graph, context);
    if (!module) {
        stream.flush();
        return diagnostics.empty() ? "invalid lattice MLIR" : diagnostics;
    }
    return "";
}

void validate_lattice_mlir(std::string const& graph) {
    auto error = parse_error_message(graph);
    if (!error.empty()) {
        throw nb::value_error(error.c_str());
    }
}

nb::dict lattice_mlir_status(std::string const& graph) {
    nb::dict out;
    auto error = parse_error_message(graph);
    out["valid"] = error.empty();
    out["diagnostics"] = error;
    return out;
}

std::vector<std::string>
lattice_mlir_operation_names(std::string const& graph) {
    return with_verified_module(graph, [](mlir::ModuleOp module) {
        auto functions = module.getOps<mlir::func::FuncOp>();
        auto function = *functions.begin();
        std::vector<std::string> names;
        for (auto& op : function.front()) {
            auto name = op.getName();
            if (name.getDialectNamespace() == "lattice") {
                names.emplace_back(name.getStringRef().str());
            }
        }
        return names;
    });
}

std::string type_to_string(mlir::Type type) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    type.print(stream);
    stream.flush();
    return text;
}

nb::object attribute_to_python(mlir::Attribute attr) {
    if (auto value = mlir::dyn_cast<mlir::StringAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
        return nb::str(value.getLeafReference().str().c_str());
    }
    if (auto value = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
        nb::list out;
        for (auto item : value.asArrayRef()) {
            out.append(nb::int_(item));
        }
        return out;
    }
    if (auto value = mlir::dyn_cast<mlir::DenseF64ArrayAttr>(attr)) {
        nb::list out;
        for (auto item : value.asArrayRef()) {
            out.append(nb::float_(item));
        }
        return out;
    }
    if (auto value = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
        return nb::float_(value.getValueAsDouble());
    }
    if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        return nb::int_(value.getInt());
    }
    if (auto value = mlir::dyn_cast<lattice::CoordAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::WeightLayoutAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::ActivationAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::GeluApproxAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::JoinAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::BinaryOpAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::PoolModeAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::VoxelReductionAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::PointInterpolationAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::PackingAttr>(attr)) {
        nb::dict out;
        out["kind"] = value.getKind().str();
        out["group_size"] = value.getGroupSize();
        if (value.getScaleType()) {
            std::string type_text;
            llvm::raw_string_ostream stream(type_text);
            value.getScaleType().print(stream);
            stream.flush();
            out["scale_dtype"] = type_text;
        }
        out["mode"] = value.getMode().str();
        return out;
    }
    std::string text;
    llvm::raw_string_ostream stream(text);
    attr.print(stream);
    stream.flush();
    return nb::str(text.c_str());
}

std::vector<std::string>
module_string_array(mlir::ModuleOp module, llvm::StringRef name) {
    std::vector<std::string> out;
    auto attr = module->getAttrOfType<mlir::ArrayAttr>(name);
    if (!attr) {
        return out;
    }
    out.reserve(attr.size());
    for (auto item : attr) {
        auto string = mlir::dyn_cast<mlir::StringAttr>(item);
        if (string) {
            out.emplace_back(string.getValue().str());
        }
    }
    return out;
}

nb::list strings_to_list(std::vector<std::string> const& values) {
    nb::list out;
    for (auto const& value : values) {
        out.append(nb::str(value.c_str()));
    }
    return out;
}

nb::dict lattice_mlir_schema() {
    nb::dict out;
    out["types"] = strings_to_list({
        lattice::SparseTensorType::getMnemonic().str(),
        lattice::WeightType::getMnemonic().str(),
    });
    out["attrs"] = strings_to_list({
        lattice::CoordAttr::getMnemonic().str(),
        lattice::FeatureLayoutAttr::getMnemonic().str(),
        lattice::WeightLayoutAttr::getMnemonic().str(),
        lattice::PackingAttr::getMnemonic().str(),
        lattice::ActivationAttr::getMnemonic().str(),
        lattice::GeluApproxAttr::getMnemonic().str(),
        lattice::JoinAttr::getMnemonic().str(),
        lattice::BinaryOpAttr::getMnemonic().str(),
        lattice::PoolModeAttr::getMnemonic().str(),
        lattice::VoxelReductionAttr::getMnemonic().str(),
        lattice::PointInterpolationAttr::getMnemonic().str(),
    });
    out["ops"] = strings_to_list({
        lattice::WeightOp::getOperationName().str(),
        lattice::SparseMakeOp::getOperationName().str(),
        lattice::SparseDecomposeOp::getOperationName().str(),
        lattice::SparseWithFeaturesOp::getOperationName().str(),
        lattice::Conv3DOp::getOperationName().str(),
        lattice::SubmConv3DOp::getOperationName().str(),
        lattice::TargetConv3DOp::getOperationName().str(),
        lattice::ConvTranspose3DOp::getOperationName().str(),
        lattice::GenerativeConvTranspose3DOp::getOperationName().str(),
        lattice::Pool3DOp::getOperationName().str(),
        lattice::GlobalPoolOp::getOperationName().str(),
        lattice::VoxelizeOp::getOperationName().str(),
        lattice::DevoxelizeOp::getOperationName().str(),
        lattice::LinearOp::getOperationName().str(),
        lattice::ActivationOp::getOperationName().str(),
        lattice::BatchNormOp::getOperationName().str(),
        lattice::LayerNormOp::getOperationName().str(),
        lattice::RMSNormOp::getOperationName().str(),
        lattice::SparseBinaryOp::getOperationName().str(),
    });
    out["schema_digest"] = lattice::kArtifactSchemaDigest.str();
    return out;
}

nb::dict lattice_mlir_plan(std::string const& graph) {
    return with_verified_module(graph, [](mlir::ModuleOp module) {
        auto functions = module.getOps<mlir::func::FuncOp>();
        auto function = *functions.begin();

        llvm::DenseMap<mlir::Value, std::string> valueNames;
        auto label = [&](mlir::Value value) -> std::string {
            auto found = valueNames.find(value);
            if (found != valueNames.end()) {
                return found->second;
            }
            auto next = "v" + std::to_string(valueNames.size());
            valueNames[value] = next;
            return next;
        };

        nb::list args;
        auto& entry = function.front();
        auto inputNames = module_string_array(module, "lattice.input_names");
        auto inputRoles = module_string_array(module, "lattice.input_roles");
        for (auto argument : entry.getArguments()) {
            auto name = "arg" + std::to_string(argument.getArgNumber());
            valueNames[argument] = name;
            nb::dict arg;
            arg["name"] = name;
            arg["abi_name"] = inputNames.at(argument.getArgNumber());
            arg["type"] = type_to_string(argument.getType());
            arg["role"] = inputRoles.at(argument.getArgNumber());
            args.append(arg);
        }

        nb::list ops;
        nb::list returns;
        nb::list outputs;
        auto outputNames = module_string_array(module, "lattice.output_names");
        auto outputRoles = module_string_array(module, "lattice.output_roles");
        for (auto& op : entry) {
            if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
                auto index = 0U;
                for (auto operand : returnOp.getOperands()) {
                    auto value = label(operand);
                    returns.append(value);
                    nb::dict output;
                    output["name"] = value;
                    output["abi_name"] = outputNames.at(index);
                    output["type"] = type_to_string(operand.getType());
                    output["role"] = outputRoles.at(index);
                    outputs.append(output);
                    ++index;
                }
                continue;
            }
            auto opName = op.getName();
            if (opName.getDialectNamespace() != "lattice") {
                throw nb::value_error(
                    "lattice MLIR plan contains a non-lattice operation"
                );
            }

            nb::dict item;
            item["name"] = opName.getStringRef().str();

            nb::list operands;
            nb::list operandTypes;
            for (auto operand : op.getOperands()) {
                operands.append(label(operand));
                operandTypes.append(type_to_string(operand.getType()));
            }
            item["operands"] = operands;
            item["operand_types"] = operandTypes;

            nb::list results;
            nb::list resultTypes;
            for (auto result : op.getResults()) {
                auto name = "v" + std::to_string(valueNames.size());
                valueNames[result] = name;
                results.append(name);
                resultTypes.append(type_to_string(result.getType()));
            }
            item["results"] = results;
            item["result_types"] = resultTypes;

            nb::dict attrs;
            for (auto attr : op.getAttrs()) {
                attrs[nb::str(attr.getName().str().c_str())] =
                    attribute_to_python(attr.getValue());
            }
            item["attrs"] = attrs;
            ops.append(item);
        }

        nb::dict out;
        auto irVersionAttr =
            module->getAttrOfType<mlir::IntegerAttr>("lattice.ir_version");
        auto weightFileAttr =
            module->getAttrOfType<mlir::StringAttr>("lattice.weight_file");
        auto schemaDigestAttr =
            module->getAttrOfType<mlir::StringAttr>("lattice.schema_digest");
        out["ir_version"] = irVersionAttr.getInt();
        out["schema_digest"] = schemaDigestAttr.getValue().str();
        out["weight_file"] = weightFileAttr.getValue().str();
        out["name"] = function.getSymName().str();
        out["args"] = args;
        out["ops"] = ops;
        out["returns"] = returns;
        out["outputs"] = outputs;
        return out;
    });
}

} // namespace

void register_mlir(nb::module_& module) {
    module.def(
        "validate_lattice_mlir",
        &validate_lattice_mlir,
        nb::sig("def validate_lattice_mlir(graph: str) -> None"),
        "Parse and verify lattice MLIR with the native MLIR dialect."
    );
    module.def(
        "lattice_mlir_status",
        &lattice_mlir_status,
        nb::sig("def lattice_mlir_status(graph: str) -> dict[str, object]"),
        "Return native MLIR parse/verify status for a lattice graph."
    );
    module.def(
        "lattice_mlir_operation_names",
        &lattice_mlir_operation_names,
        nb::sig("def lattice_mlir_operation_names(graph: str) -> list[str]"),
        "Return lattice operation names from a parsed and verified MLIR graph."
    );
    module.def(
        "lattice_mlir_schema",
        &lattice_mlir_schema,
        nb::sig("def lattice_mlir_schema() -> dict[str, object]"),
        "Return native MLIR lattice dialect surface metadata."
    );
    module.def(
        "lattice_mlir_plan",
        &lattice_mlir_plan,
        nb::sig("def lattice_mlir_plan(graph: str) -> dict[str, object]"),
        "Return a structured runtime plan from parsed and verified lattice "
        "MLIR."
    );
}

} // namespace mlx_lattice::bindings

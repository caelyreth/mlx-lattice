#include "bindings/registrations.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

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

std::string parse_error_message(std::string const& graph) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, lattice::LatticeDialect>();

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

    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(graph), &context
    );
    if (!module) {
        stream.flush();
        return diagnostics.empty() ? "failed to parse lattice MLIR"
                                   : diagnostics;
    }
    if (mlir::failed(module->verify())) {
        stream.flush();
        return diagnostics.empty() ? "failed to verify lattice MLIR"
                                   : diagnostics;
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
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, lattice::LatticeDialect>();

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

    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(graph), &context
    );
    if (!module || mlir::failed(module->verify())) {
        stream.flush();
        throw nb::value_error(
            diagnostics.empty() ? "invalid lattice MLIR" : diagnostics.c_str()
        );
    }

    std::vector<std::string> names;
    module->walk([&](mlir::Operation* op) {
        auto name = op->getName();
        if (name.getDialectNamespace() == "lattice") {
            names.emplace_back(name.getStringRef().str());
        }
    });
    return names;
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
    if (auto value = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
        return nb::float_(value.getValueAsDouble());
    }
    if (auto value = mlir::dyn_cast<lattice::CoordAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::WeightLayoutAttr>(attr)) {
        return nb::str(value.getValue().str().c_str());
    }
    if (auto value = mlir::dyn_cast<lattice::JoinAttr>(attr)) {
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

nb::dict lattice_mlir_plan(std::string const& graph) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, lattice::LatticeDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    std::string diagnostics;
    llvm::raw_string_ostream diagnosticStream(diagnostics);
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic& diagnostic) {
            diagnostic.print(diagnosticStream);
            diagnosticStream << "\n";
        }
    );

    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(graph), &context
    );
    if (!module || mlir::failed(module->verify())) {
        diagnosticStream.flush();
        throw nb::value_error(
            diagnostics.empty() ? "invalid lattice MLIR" : diagnostics.c_str()
        );
    }

    auto functions = module->getOps<mlir::func::FuncOp>();
    auto functionIt = functions.begin();
    if (functionIt == functions.end()) {
        throw nb::value_error("lattice MLIR module has no func.func");
    }
    auto function = *functionIt;
    if (++functionIt != functions.end()) {
        throw nb::value_error(
            "lattice MLIR runtime import expects exactly one func.func"
        );
    }
    if (function.empty()) {
        throw nb::value_error("lattice MLIR function has no entry block");
    }

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
    for (auto argument : entry.getArguments()) {
        auto name = "arg" + std::to_string(argument.getArgNumber());
        valueNames[argument] = name;
        nb::dict arg;
        arg["name"] = name;
        args.append(arg);
    }

    nb::list ops;
    nb::list returns;
    for (auto& op : entry) {
        if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
            for (auto operand : returnOp.getOperands()) {
                returns.append(label(operand));
            }
            continue;
        }
        auto opName = op.getName();
        if (opName.getDialectNamespace() != "lattice") {
            continue;
        }

        nb::dict item;
        item["name"] = opName.getStringRef().str();

        nb::list operands;
        for (auto operand : op.getOperands()) {
            operands.append(label(operand));
        }
        item["operands"] = operands;

        nb::list results;
        for (auto result : op.getResults()) {
            auto name = "v" + std::to_string(valueNames.size());
            valueNames[result] = name;
            results.append(name);
        }
        item["results"] = results;

        nb::dict attrs;
        for (auto attr : op.getAttrs()) {
            attrs[nb::str(attr.getName().str().c_str())] =
                attribute_to_python(attr.getValue());
        }
        item["attrs"] = attrs;
        ops.append(item);
    }

    nb::dict out;
    out["name"] = function.getSymName().str();
    out["args"] = args;
    out["ops"] = ops;
    out["returns"] = returns;
    return out;
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
        "lattice_mlir_plan",
        &lattice_mlir_plan,
        nb::sig("def lattice_mlir_plan(graph: str) -> dict[str, object]"),
        "Return a structured runtime plan from parsed and verified lattice "
        "MLIR."
    );
}

} // namespace mlx_lattice::bindings

#include "Lattice/Dialect/Lattice/IR/LatticeDialect.h"

#include "Lattice/Dialect/Lattice/IR/LatticeAttrs.h"
#include "Lattice/Dialect/Lattice/IR/LatticeOps.h"
#include "Lattice/Dialect/Lattice/IR/LatticeTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace lattice;

#define GET_ATTRDEF_CLASSES
#include "Lattice/Dialect/Lattice/IR/LatticeAttrs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "Lattice/Dialect/Lattice/IR/LatticeDialect.cpp.inc"
#include "Lattice/Dialect/Lattice/IR/LatticeTypes.cpp.inc"

namespace {

ParseResult parseKeywordLike(AsmParser& parser, std::string& value) {
    return parser.parseKeywordOrString(&value);
}

void printKeywordLike(AsmPrinter& printer, StringRef value) {
    printer.printKeywordOrString(value);
}

template <typename AttrT> Attribute parseSingleStringAttr(AsmParser& parser) {
    std::string value;
    if (parser.parseLess() || parseKeywordLike(parser, value) ||
        parser.parseGreater()) {
        return {};
    }
    return AttrT::get(parser.getContext(), value);
}

template <typename AttrT>
void printSingleStringAttr(AsmPrinter& printer, AttrT attr) {
    printer << "<";
    printKeywordLike(printer, attr.getValue());
    printer << ">";
}

} // namespace

Attribute CoordAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<CoordAttr>(parser);
}

void CoordAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute FeatureLayoutAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<FeatureLayoutAttr>(parser);
}

void FeatureLayoutAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute WeightLayoutAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<WeightLayoutAttr>(parser);
}

void WeightLayoutAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute ActivationAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<ActivationAttr>(parser);
}

void ActivationAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute GeluApproxAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<GeluApproxAttr>(parser);
}

void GeluApproxAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute JoinAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<JoinAttr>(parser);
}

void JoinAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute BinaryOpAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<BinaryOpAttr>(parser);
}

void BinaryOpAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute PoolModeAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<PoolModeAttr>(parser);
}

void PoolModeAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute VoxelReductionAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<VoxelReductionAttr>(parser);
}

void VoxelReductionAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute PointInterpolationAttr::parse(AsmParser& parser, Type) {
    return parseSingleStringAttr<PointInterpolationAttr>(parser);
}

void PointInterpolationAttr::print(AsmPrinter& printer) const {
    printSingleStringAttr(printer, *this);
}

Attribute PackingAttr::parse(AsmParser& parser, Type) {
    std::string kind;
    if (parser.parseLess() || parseKeywordLike(parser, kind)) {
        return {};
    }

    if (kind == "dense") {
        if (parser.parseGreater()) {
            return {};
        }
        return PackingAttr::get(parser.getContext(), kind, 0, Type(), "");
    }

    unsigned groupSize = 0;
    Type scaleType;
    std::string mode;
    if (parser.parseComma() || parser.parseKeyword("group_size") ||
        parser.parseEqual() || parser.parseInteger(groupSize) ||
        parser.parseComma() || parser.parseKeyword("scale_dtype") ||
        parser.parseEqual() || parser.parseType(scaleType) ||
        parser.parseComma() || parser.parseKeyword("mode") ||
        parser.parseEqual() || parseKeywordLike(parser, mode) ||
        parser.parseGreater()) {
        return {};
    }
    return PackingAttr::get(
        parser.getContext(), kind, groupSize, scaleType, mode
    );
}

void PackingAttr::print(AsmPrinter& printer) const {
    printer << "<";
    printKeywordLike(printer, getKind());
    if (getKind() != "dense") {
        printer << ", group_size = " << getGroupSize() << ", scale_dtype = ";
        printer.printType(getScaleType());
        printer << ", mode = ";
        printKeywordLike(printer, getMode());
    }
    printer << ">";
}

Type SparseTensorType::parse(AsmParser& parser) {
    unsigned rank = 0;
    std::string coord;
    std::string feature;
    Type dtype;
    if (parser.parseLess() || parser.parseKeyword("rank") ||
        parser.parseEqual() || parser.parseInteger(rank) ||
        parser.parseComma() || parser.parseKeyword("coord") ||
        parser.parseEqual() || parseKeywordLike(parser, coord) ||
        parser.parseComma() || parser.parseKeyword("feature") ||
        parser.parseEqual() || parseKeywordLike(parser, feature) ||
        parser.parseComma() || parser.parseKeyword("dtype") ||
        parser.parseEqual() || parser.parseType(dtype) ||
        parser.parseGreater()) {
        return {};
    }
    return SparseTensorType::get(
        parser.getContext(), rank, coord, feature, dtype
    );
}

void SparseTensorType::print(AsmPrinter& printer) const {
    printer << "<rank = " << getRank() << ", coord = ";
    printKeywordLike(printer, getCoord());
    printer << ", feature = ";
    printKeywordLike(printer, getFeature());
    printer << ", dtype = ";
    printer.printType(getDtype());
    printer << ">";
}

Type WeightType::parse(AsmParser& parser) {
    std::string family;
    Type dtype;
    if (parser.parseLess() || parseKeywordLike(parser, family) ||
        parser.parseComma() || parser.parseType(dtype) ||
        parser.parseGreater()) {
        return {};
    }
    return WeightType::get(parser.getContext(), family, dtype);
}

void WeightType::print(AsmPrinter& printer) const {
    printer << "<";
    printKeywordLike(printer, getFamily());
    printer << ", ";
    printer.printType(getDtype());
    printer << ">";
}

void LatticeDialect::initialize() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "Lattice/Dialect/Lattice/IR/LatticeAttrs.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "Lattice/Dialect/Lattice/IR/LatticeTypes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "Lattice/Dialect/Lattice/IR/LatticeOps.cpp.inc"
        >();
}

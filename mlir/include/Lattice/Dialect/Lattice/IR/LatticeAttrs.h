#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

#define GET_ATTRDEF_CLASSES
#include "Lattice/Dialect/Lattice/IR/LatticeAttrs.h.inc"

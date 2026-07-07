#pragma once

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Lattice/Dialect/Lattice/IR/LatticeAttrs.h"
#include "Lattice/Dialect/Lattice/IR/LatticeDialect.h"
#include "Lattice/Dialect/Lattice/IR/LatticeTypes.h"

#define GET_OP_CLASSES
#include "Lattice/Dialect/Lattice/IR/LatticeOps.h.inc"

set(LLVM_TARGET_DEFINITIONS LatticeOps.td)

mlir_tablegen(LatticeDialect.h.inc -gen-dialect-decls -dialect=lattice)
mlir_tablegen(LatticeDialect.cpp.inc -gen-dialect-defs -dialect=lattice)
mlir_tablegen(LatticeOps.h.inc -gen-op-decls)
mlir_tablegen(LatticeOps.cpp.inc -gen-op-defs)
mlir_tablegen(LatticeTypes.h.inc -gen-typedef-decls -typedefs-dialect=lattice)
mlir_tablegen(LatticeTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=lattice)
mlir_tablegen(LatticeAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=lattice)
mlir_tablegen(LatticeAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=lattice)

add_public_tablegen_target(MLIRLatticeIncGen)

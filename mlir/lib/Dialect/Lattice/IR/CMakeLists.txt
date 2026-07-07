add_mlir_dialect_library(MLIRLatticeDialect
  LatticeDialect.cpp
  LatticeOps.cpp

  DEPENDS
  MLIRLatticeIncGen
)

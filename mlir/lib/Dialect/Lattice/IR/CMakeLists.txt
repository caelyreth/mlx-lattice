add_mlir_dialect_library(MLIRLatticeDialect
  LatticeArtifact.cpp
  LatticeDialect.cpp
  LatticeOps.cpp

  DEPENDS
  MLIRLatticeIncGen
)

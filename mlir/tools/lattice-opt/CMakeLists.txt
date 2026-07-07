add_mlir_tool(lattice-opt
  lattice-opt.cpp
)

if(TARGET MLIR)
  target_link_libraries(lattice-opt
    PRIVATE
    MLIRLatticeDialect
    MLIR
  )
else()
  target_link_libraries(lattice-opt
    PRIVATE
    MLIRFuncDialect
    MLIRIR
    MLIRLatticeDialect
    MLIRMlirOptMain
  )
endif()

#ifndef LATTICE_DIALECT_LATTICE_IR_LATTICEDIALECT_TD
#define LATTICE_DIALECT_LATTICE_IR_LATTICEDIALECT_TD

include "mlir/IR/OpBase.td"

def Lattice_Dialect : Dialect {
  let name = "lattice";
  let cppNamespace = "::lattice";
  let summary = "Portable sparse lattice model artifact dialect";
  let description = [{
    The lattice dialect represents backend-neutral sparse lattice model
    semantics. It is the artifact contract between source runtimes such as
    Torch/CUDA training stacks and target runtimes such as MLX/Metal inference.

    The dialect intentionally excludes runtime object identity, coordinate
    manager handles, backend execution views, CSR layouts, implicit-GEMM maps,
    TensorOps route names, and kernel dispatch choices.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // LATTICE_DIALECT_LATTICE_IR_LATTICEDIALECT_TD

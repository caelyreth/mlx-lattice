# Lattice MLIR Dialect

This directory contains the backend-neutral MLIR dialect source for portable
sparse lattice model artifacts.

The dialect is optional developer tooling. It is not part of the default
`mlx-lattice` wheel build and should not make normal MLX runtime installs
depend on LLVM/MLIR.

Build opt-in:

```sh
cmake -S . -B build/mlir -DSKBUILD=ON -DMLX_LATTICE_ENABLE_MLIR=ON
cmake --build build/mlir --target lattice-opt
```

Validate committed dialect fixtures:

```sh
build/mlir/mlir/tools/lattice-opt/lattice-opt \
  --lattice-verify-artifact \
  mlir/test/Dialect/Lattice/valid/conv3d_basic.mlir
```

The first slice intentionally owns only the portable contract:

- sparse value ABI;
- weight binding;
- sparse construction/decomposition;
- feature-only sparse identity update;
- forward/submanifold/target/transpose/generative sparse convolution;
- local and global sparse pooling;
- point/voxel conversion;
- generic coordinate-aligned sparse binary algebra.
- artifact ABI metadata validation for `lattice.ir_version`,
  `lattice.schema_digest`, `lattice.weight_file`, and explicit model
  input/output ABI arrays.
- importer-facing argument/output type, public ABI name, and sparse component
  role metadata for the verified entry function.

It does not contain MLX importer code, Torch exporter code, TensorOps/CSR
execution views, or JSON artifact compatibility.

## Source-of-truth rule

Dialect semantics should live in TableGen/C++ verifier definitions here.
Python metadata in `lattice-contract` may later be generated or mirrored from
these definitions, but it should not become a second graph IR source of truth.

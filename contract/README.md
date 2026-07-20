# lattice-contract

`lattice-contract` is the backend-neutral artifact contract package used by
`mlx-lattice` and future training-side exporters.

The forward contract is MLIR-first. Dialect declarations live in one annotated
Python schema and generate the textual MLIR builder surface at import time, so
exporters do not hand-maintain op-name and attribute-name maps.

It intentionally does not import MLX, Torch, native extensions, or backend
runtime objects. The exchange target is `graph.mlir + weights.safetensors`;
shallow package metadata can be added later, but graph semantics belong in
MLIR.

The package exports the artifact ABI constants used by builders and importers:
`CURRENT_DIALECT_VERSION`, `DIALECT_SCHEMA_DIGEST`, `ARTIFACT_GRAPH_FILE`,
and `ARTIFACT_WEIGHT_FILE`.

The current contract is IR v2. Sparse ABI coordinates are physical positions
and must be exactly divisible by the sparse stride. Portable convolution
artifacts declare `accumulation = "canonical_f32"`; FP16 may be used for tensor
storage but not to alter the graph's accumulation semantics.

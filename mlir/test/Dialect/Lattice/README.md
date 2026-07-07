# Lattice dialect fixtures

These files are the tracked verifier/importer fixtures for the first lattice
MLIR slice.

They intentionally mirror the ignored design examples under
`references/mlir_examples/`. The optional `lattice-opt` developer tool can
parse and verify them when `MLX_LATTICE_ENABLE_MLIR=ON`.

The invalid files currently validate failure coverage by exit status. Once the
MLIR tree grows a lit configuration, they should become `-verify-diagnostics`
tests with stable expected diagnostics.

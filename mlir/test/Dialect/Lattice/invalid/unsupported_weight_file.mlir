// Invalid: the current artifact ABI uses a fixed safetensors payload name.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "e48cb610f907d8c7afbe66c197f2e01ab7ba3519a3f3d452b9643768f5c476c9",
  lattice.input_names = [],
  lattice.input_roles = [],
  lattice.output_names = [],
  lattice.output_roles = [],
  lattice.weight_file = "model.safetensors"
} {
  func.func @forward() {
    return
  }
}

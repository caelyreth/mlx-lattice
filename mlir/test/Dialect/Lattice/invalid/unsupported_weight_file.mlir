// Invalid: the current artifact ABI uses a fixed safetensors payload name.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "33a97d62e5b150b98940c62284f42b326e879cc4aca2747cdbc0d77c851f66c7",
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

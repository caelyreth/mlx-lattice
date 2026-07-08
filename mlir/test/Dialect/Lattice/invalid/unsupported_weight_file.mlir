// Invalid: the current artifact ABI uses a fixed safetensors payload name.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "314833e397548364385e5a24c1faf5ebcd4eadc3a0d750a0bed444e2c855c4a1",
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

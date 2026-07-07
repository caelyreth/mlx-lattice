// Invalid: the current artifact ABI uses a fixed safetensors payload name.
module attributes {
  lattice.ir_version = 0,
  lattice.weight_file = "model.safetensors"
} {
  func.func @forward() {
    return
  }
}

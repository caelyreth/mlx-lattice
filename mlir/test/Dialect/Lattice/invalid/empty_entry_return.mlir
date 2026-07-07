// Invalid: executable artifacts must return at least one value.
module attributes {
  lattice.ir_version = 0,
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward() {
    return
  }
}

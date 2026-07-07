// Invalid: only lattice.ir_version = 0 is currently accepted.
module attributes {
  lattice.ir_version = 1,
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward() {
    return
  }
}

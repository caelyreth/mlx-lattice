// Invalid: lattice artifact modules must declare the dialect ABI version.
module attributes {
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward() {
    return
  }
}

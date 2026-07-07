// Invalid: lattice artifact modules must declare the weight payload file.
module attributes {
  lattice.ir_version = 0
} {
  func.func @forward() {
    return
  }
}

// Invalid: the executable artifact entry point is named @forward.
module attributes {
  lattice.ir_version = 0,
  lattice.weight_file = "weights.safetensors"
} {
  func.func @predict(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {
    return %x : tensor<?x4xi32>
  }
}

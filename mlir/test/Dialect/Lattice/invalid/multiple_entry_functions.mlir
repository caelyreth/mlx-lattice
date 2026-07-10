// Invalid: executable artifacts must not contain multiple functions.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "8a5ace10e29b47304594c1b66608ab64318c68568a69f4dcbc1ed8c570d73088",
  lattice.input_names = ["input0"],
  lattice.input_roles = ["tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {
    return %x : tensor<?x4xi32>
  }

  func.func @aux(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {
    return %x : tensor<?x4xi32>
  }
}

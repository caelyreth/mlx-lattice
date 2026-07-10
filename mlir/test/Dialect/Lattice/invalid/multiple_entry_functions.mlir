// Invalid: executable artifacts must not contain multiple functions.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "33a97d62e5b150b98940c62284f42b326e879cc4aca2747cdbc0d77c851f66c7",
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

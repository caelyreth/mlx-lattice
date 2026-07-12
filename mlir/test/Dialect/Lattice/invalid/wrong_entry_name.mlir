// Invalid: the executable artifact entry point is named @forward.
module attributes {
  lattice.ir_version = 1,
  lattice.schema_digest = "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa",
  lattice.input_names = [],
  lattice.input_roles = [],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @predict(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {
    return %x : tensor<?x4xi32>
  }
}

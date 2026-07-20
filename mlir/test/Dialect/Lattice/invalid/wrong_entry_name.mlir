// Invalid: the executable artifact entry point is named @forward.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
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

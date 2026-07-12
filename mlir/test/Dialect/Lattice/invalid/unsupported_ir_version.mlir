// Invalid: only lattice.ir_version = 1 is currently accepted.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa",
  lattice.input_names = [],
  lattice.input_roles = [],
  lattice.output_names = [],
  lattice.output_roles = [],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward() {
    return
  }
}

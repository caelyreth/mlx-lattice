// Invalid: executable artifacts must return at least one value.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "81c8424987d97d0f6cd514b50d8db2307e467e55f32a6c30dfdf6e311d565443",
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

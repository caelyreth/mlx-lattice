// Invalid: executable artifact bodies may only contain lattice ops and return.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "81c8424987d97d0f6cd514b50d8db2307e467e55f32a6c30dfdf6e311d565443",
  lattice.input_names = ["input0"],
  lattice.input_roles = ["tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {
    %out = func.call @forward(%x) : (tensor<?x4xi32>) -> tensor<?x4xi32>
    return %out : tensor<?x4xi32>
  }
}

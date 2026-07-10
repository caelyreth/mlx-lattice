// Invalid: executable artifact bodies may only contain lattice ops and return.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "e48cb610f907d8c7afbe66c197f2e01ab7ba3519a3f3d452b9643768f5c476c9",
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

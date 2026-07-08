// Invalid: executable artifact bodies may only contain lattice ops and return.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "314833e397548364385e5a24c1faf5ebcd4eadc3a0d750a0bed444e2c855c4a1",
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

// Invalid: executable artifact bodies may only contain lattice ops and return.
module attributes {
  lattice.ir_version = 0,
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%x: tensor<?x4xi32>) -> tensor<?x4xi32> {
    %out = func.call @forward(%x) : (tensor<?x4xi32>) -> tensor<?x4xi32>
    return %out : tensor<?x4xi32>
  }
}

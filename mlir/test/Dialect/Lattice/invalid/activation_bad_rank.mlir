// Invalid: feature activations operate on rank-2 feature tensors.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "de8cda6380a1e82a3ba08d215a77a43a0a7088d74e81dbc2afa2446dbb79bfd1",
  lattice.input_names = ["input0"],
  lattice.input_roles = ["tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%features: tensor<?x16x32xf16>) -> tensor<?x16x32xf16> {
    %out = lattice.activation %features
      {kind = #lattice.activation<relu>,
       approximate = #lattice.gelu_approx<none>,
       alpha = 0.01 : f32,
       beta = 1.0 : f32,
       threshold = 20.0 : f32}
      : (tensor<?x16x32xf16>) -> tensor<?x16x32xf16>
    return %out : tensor<?x16x32xf16>
  }
}

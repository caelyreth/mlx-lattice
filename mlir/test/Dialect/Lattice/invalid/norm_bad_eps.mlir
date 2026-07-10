// Invalid: normalization epsilon must be positive.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "de8cda6380a1e82a3ba08d215a77a43a0a7088d74e81dbc2afa2446dbb79bfd1",
  lattice.input_names = ["input0", "input1"],
  lattice.input_roles = ["tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%features: tensor<?x32xf16>,
                     %scale: !lattice.weight<channel, f16>)
      -> tensor<?x32xf16> {
    %out = lattice.rms_norm %features, %scale
      {eps = 0.0 : f32}
      : (tensor<?x32xf16>, !lattice.weight<channel, f16>)
        -> tensor<?x32xf16>
    return %out : tensor<?x32xf16>
  }
}

// Invalid: normalization scale/mean/var must use channel-family weights.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "8a5ace10e29b47304594c1b66608ab64318c68568a69f4dcbc1ed8c570d73088",
  lattice.input_names = ["input0", "input1", "input2"],
  lattice.input_roles = ["tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%features: tensor<?x32xf16>,
                     %bad_scale: !lattice.weight<linear, f16>,
                     %bias: !lattice.weight<bias, f16>)
      -> tensor<?x32xf16> {
    %out = lattice.layer_norm %features, %bad_scale, %bias
      {eps = 1.0e-05 : f32}
      : (tensor<?x32xf16>,
         !lattice.weight<linear, f16>,
         !lattice.weight<bias, f16>)
        -> tensor<?x32xf16>
    return %out : tensor<?x32xf16>
  }
}

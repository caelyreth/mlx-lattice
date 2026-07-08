// Invalid: normalization scale/mean/var must use channel-family weights.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "314833e397548364385e5a24c1faf5ebcd4eadc3a0d750a0bed444e2c855c4a1",
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

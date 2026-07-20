// Invalid: normalization scale/mean/var must use channel-family weights.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
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

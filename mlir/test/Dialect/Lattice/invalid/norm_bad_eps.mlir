// Invalid: normalization epsilon must be positive.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
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

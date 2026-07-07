// Invalid: normalization epsilon must be positive.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "81c8424987d97d0f6cd514b50d8db2307e467e55f32a6c30dfdf6e311d565443",
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

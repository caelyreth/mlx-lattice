// Invalid: normalization epsilon must be positive.
module attributes {
  lattice.ir_version = 0,
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

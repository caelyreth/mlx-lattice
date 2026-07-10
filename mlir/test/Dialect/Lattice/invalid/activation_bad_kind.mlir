// Invalid: activation kind is part of the stable ABI enum.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "33a97d62e5b150b98940c62284f42b326e879cc4aca2747cdbc0d77c851f66c7",
  lattice.input_names = ["input0"],
  lattice.input_roles = ["tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%features: tensor<?x32xf16>) -> tensor<?x32xf16> {
    %out = lattice.activation %features
      {kind = #lattice.activation<mish>,
       approximate = #lattice.gelu_approx<none>,
       alpha = 0.01 : f32,
       beta = 1.0 : f32,
       threshold = 20.0 : f32}
      : (tensor<?x32xf16>) -> tensor<?x32xf16>
    return %out : tensor<?x32xf16>
  }
}

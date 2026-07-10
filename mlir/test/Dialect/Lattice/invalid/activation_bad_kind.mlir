// Invalid: activation kind is part of the stable ABI enum.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "8a5ace10e29b47304594c1b66608ab64318c68568a69f4dcbc1ed8c570d73088",
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

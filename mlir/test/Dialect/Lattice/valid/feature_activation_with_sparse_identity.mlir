// Valid: dense activation updates feature rows while preserving sparse support explicitly.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
  lattice.input_names = ["input0", "input1", "input2"],
  lattice.input_roles = ["tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %coords: tensor<?x4xi32>,
    %features: tensor<?x32xf16>,
    %active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3,
                              coord = batch_x_y_z,
                              feature = row_channel,
                              dtype = f16> {
    %input = lattice.sparse.make %coords, %features, %active
      {stride = array<i64: 1, 1, 1>,
       coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3,
                                  coord = batch_x_y_z,
                                  feature = row_channel,
                                  dtype = f16>

    %coords0, %features0, %active0 = lattice.sparse.decompose %input
      : !lattice.sparse_tensor<rank = 3,
                               coord = batch_x_y_z,
                               feature = row_channel,
                               dtype = f16>
        -> (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)

    %activated = lattice.activation %features0
      {kind = #lattice.activation<gelu>,
       approximate = #lattice.gelu_approx<tanh>,
       alpha = 0.01 : f32,
       beta = 1.0 : f32,
       threshold = 20.0 : f32}
      : (tensor<?x32xf16>) -> tensor<?x32xf16>

    %out = lattice.sparse.with_features %input, %activated
      : (!lattice.sparse_tensor<rank = 3,
                                coord = batch_x_y_z,
                                feature = row_channel,
                                dtype = f16>,
         tensor<?x32xf16>)
        -> !lattice.sparse_tensor<rank = 3,
                                  coord = batch_x_y_z,
                                  feature = row_channel,
                                  dtype = f16>

    return %out : !lattice.sparse_tensor<rank = 3,
                                        coord = batch_x_y_z,
                                        feature = row_channel,
                                        dtype = f16>
  }
}

// Valid: global sparse pooling reduces sparse rows to dense batch rows.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
  lattice.input_names = ["input0", "input1", "input2"],
  lattice.input_roles = ["tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%coords: tensor<?x4xi32>,
                     %features: tensor<?x32xf32>,
                     %active: tensor<1xi32>)
      -> tensor<?x32xf32> {
    %input = lattice.sparse.make %coords, %features, %active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x32xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    %out = lattice.global_pool %input
      {mode = #lattice.pool_mode<sum>, batch_size = 2}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f32>)
        -> tensor<?x32xf32>
    return %out : tensor<?x32xf32>
  }
}

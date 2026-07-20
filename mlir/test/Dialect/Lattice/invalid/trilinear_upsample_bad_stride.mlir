// Invalid: trilinear upsampling stride must be positive.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
  lattice.input_names = ["coords", "features", "active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%coords: tensor<?x4xi32>, %features: tensor<?x8xf32>,
                     %active: tensor<1xi32>)
      -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %input = lattice.sparse.make %coords, %features, %active
      {stride = array<i64: 2, 2, 2>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x8xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %out = lattice.trilinear_upsample3d %input {stride = array<i64: 2, 0, 2>}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

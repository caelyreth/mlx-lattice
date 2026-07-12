// Valid: trilinear upsampling may consume explicit target support.
module attributes {
  lattice.ir_version = 1,
  lattice.schema_digest = "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa",
  lattice.input_names = ["source_coords", "source_features", "source_active", "target_coords", "target_features", "target_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active", "sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%source_coords: tensor<?x4xi32>,
                     %source_features: tensor<?x8xf32>,
                     %source_active: tensor<1xi32>,
                     %target_coords: tensor<?x4xi32>,
                     %target_features: tensor<?x1xf32>,
                     %target_active: tensor<1xi32>)
      -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %source = lattice.sparse.make %source_coords, %source_features, %source_active
      {stride = array<i64: 2, 2, 2>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x8xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %target = lattice.sparse.make %target_coords, %target_features, %target_active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x1xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %out = lattice.trilinear_upsample3d %source, %target
      {stride = array<i64: 2, 2, 2>}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>,
         !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

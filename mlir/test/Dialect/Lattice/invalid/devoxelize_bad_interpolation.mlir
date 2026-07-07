// Invalid: interpolation must be nearest or linear.
module attributes {
  lattice.ir_version = 0,
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%points: tensor<?x3xf32>,
                     %voxel_coords: tensor<?x4xi32>,
                     %voxel_features: tensor<?x16xf32>,
                     %voxel_active: tensor<1xi32>,
                     %batch_indices: tensor<?xi32>,
                     %point_active_rows: tensor<1xi32>)
      -> tensor<?x16xf32> {
    %voxels = lattice.sparse.make %voxel_coords, %voxel_features, %voxel_active
      {stride = array<i64: 1, 1, 1>,
       coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x16xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    %features = lattice.devoxelize
      %points, %voxels, %batch_indices, %point_active_rows
      {voxel_size = array<f64: 0.1, 0.1, 0.1>,
       origin = array<f64: 0.0, 0.0, 0.0>,
       interpolation = #lattice.point_interpolation<cubic>}
      : (tensor<?x3xf32>,
         !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f32>,
         tensor<?xi32>, tensor<1xi32>)
        -> tensor<?x16xf32>
    return %features : tensor<?x16xf32>
  }
}

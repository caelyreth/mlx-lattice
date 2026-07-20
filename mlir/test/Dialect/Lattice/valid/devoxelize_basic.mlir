// Valid: sparse voxel rows interpolate back to dense point rows.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
  lattice.input_names = ["input0", "input1", "input2", "input3", "input4", "input5"],
  lattice.input_roles = ["tensor", "tensor", "tensor", "tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
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
       interpolation = #lattice.point_interpolation<linear>}
      : (tensor<?x3xf32>,
         !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f32>,
         tensor<?xi32>, tensor<1xi32>)
        -> tensor<?x16xf32>
    return %features : tensor<?x16xf32>
  }
}

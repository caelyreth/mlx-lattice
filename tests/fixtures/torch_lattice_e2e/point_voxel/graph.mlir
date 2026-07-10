module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "545fd43029e2fbebc18404babc0ec463730a13056a18f51280d8f1a272a3369c",
  lattice.input_names = ["points", "features", "batch_indices", "active_rows"],
  lattice.input_roles = ["tensor", "tensor", "tensor", "tensor"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %points: tensor<?x3xf32>,
    %features: tensor<?x2xf32>,
    %batch_indices: tensor<?xi32>,
    %active_rows: tensor<1xi32>
  ) -> tensor<?x2xf32> {
    %voxelize = lattice.voxelize %points, %features, %batch_indices, %active_rows {voxel_size = array<f64: 1.0, 1.0, 1.0>, origin = array<f64: 0.0, 0.0, 0.0>, reduction = #lattice.voxel_reduction<mean>, stride = array<i64: 1, 1, 1>} : (tensor<?x3xf32>, tensor<?x2xf32>, tensor<?xi32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %devoxelize = lattice.devoxelize %points, %voxelize, %batch_indices, %active_rows {voxel_size = array<f64: 1.0, 1.0, 1.0>, origin = array<f64: 0.0, 0.0, 0.0>, interpolation = #lattice.point_interpolation<nearest>} : (tensor<?x3xf32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?xi32>, tensor<1xi32>) -> tensor<?x2xf32>
    return %devoxelize : tensor<?x2xf32>
  }
}

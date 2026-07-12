// Invalid: point tensor must have trailing xyz dimension 3.
module attributes {
  lattice.ir_version = 1,
  lattice.schema_digest = "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa",
  lattice.input_names = ["input0", "input1", "input2", "input3"],
  lattice.input_roles = ["tensor", "tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%points: tensor<?x4xf32>,
                     %features: tensor<?x16xf32>,
                     %batch_indices: tensor<?xi32>,
                     %active_rows: tensor<1xi32>)
      -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f32> {
    %voxels = lattice.voxelize %points, %features, %batch_indices, %active_rows
      {voxel_size = array<f64: 0.1, 0.1, 0.1>,
       origin = array<f64: 0.0, 0.0, 0.0>,
       reduction = #lattice.voxel_reduction<mean>,
       stride = array<i64: 1, 1, 1>}
      : (tensor<?x4xf32>, tensor<?x16xf32>, tensor<?xi32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    return %voxels : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                           feature = row_channel, dtype = f32>
  }
}

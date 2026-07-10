// Valid: point rows plus features produce a sparse voxel tensor.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "8a5ace10e29b47304594c1b66608ab64318c68568a69f4dcbc1ed8c570d73088",
  lattice.input_names = ["input0", "input1", "input2", "input3"],
  lattice.input_roles = ["tensor", "tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%points: tensor<?x3xf32>,
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
      : (tensor<?x3xf32>, tensor<?x16xf32>, tensor<?xi32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    return %voxels : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                           feature = row_channel, dtype = f32>
  }
}

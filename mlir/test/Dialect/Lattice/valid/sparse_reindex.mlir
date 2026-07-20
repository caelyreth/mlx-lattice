// Valid: sparse reindexing preserves exact target support and source features.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
  lattice.input_names = ["input0", "input1", "input2", "input3", "input4", "input5"],
  lattice.input_roles = ["tensor", "tensor", "tensor", "tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %source_coords: tensor<?x4xi32>,
    %source_features: tensor<?x8xf32>,
    %source_active: tensor<1xi32>,
    %target_coords: tensor<?x4xi32>,
    %target_features: tensor<?x1xf32>,
    %target_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                              feature = row_channel, dtype = f32> {
    %source = lattice.sparse.make %source_coords, %source_features, %source_active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x8xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    %target = lattice.sparse.make %target_coords, %target_features, %target_active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x1xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    %out = lattice.sparse.reindex %source, %target {fill = 0.0 : f32}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f32>,
         !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                         feature = row_channel, dtype = f32>
  }
}

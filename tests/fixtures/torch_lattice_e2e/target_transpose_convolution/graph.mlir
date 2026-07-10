module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "de8cda6380a1e82a3ba08d215a77a43a0a7088d74e81dbc2afa2446dbb79bfd1",
  lattice.input_names = ["source_coords", "source_features", "source_active", "target_coords", "target_features", "target_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active", "sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %source_coords: tensor<?x4xi32>,
    %source_features: tensor<?x2xf32>,
    %source_active: tensor<1xi32>,
    %target_coords: tensor<?x4xi32>,
    %target_features: tensor<?x1xf32>,
    %target_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %source = lattice.sparse.make %source_coords, %source_features, %source_active {stride = array<i64: 2, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x2xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %target = lattice.sparse.make %target_coords, %target_features, %target_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x1xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %up_weight = lattice.weight @up_weight {storage_key = "up.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %up_bias = lattice.weight @up_bias {storage_key = "up.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %up = lattice.target_normalized_conv_transpose3d %source, %target, %up_weight, %up_bias {kernel_size = array<i64: 3, 1, 1>, stride = array<i64: 2, 1, 1>, padding = array<i64: 1, 0, 0>, dilation = array<i64: 1, 1, 1>, eps = 0.00000001 : f32} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %up : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

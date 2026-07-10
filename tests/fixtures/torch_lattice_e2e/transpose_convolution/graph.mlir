module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "33a97d62e5b150b98940c62284f42b326e879cc4aca2747cdbc0d77c851f66c7",
  lattice.input_names = ["coords", "features", "active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %coords: tensor<?x4xi32>,
    %features: tensor<?x?xf32>,
    %active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %input = lattice.sparse.make %coords, %features, %active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x?xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %down_weight = lattice.weight @down_weight {storage_key = "down.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %down = lattice.conv3d %input, %down_weight {kernel_size = array<i64: 2, 1, 1>, stride = array<i64: 2, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %up_weight = lattice.weight @up_weight {storage_key = "up.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %up_bias = lattice.weight @up_bias {storage_key = "up.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %up = lattice.conv_transpose3d %down, %up_weight, %up_bias {kernel_size = array<i64: 2, 1, 1>, stride = array<i64: 2, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %up : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

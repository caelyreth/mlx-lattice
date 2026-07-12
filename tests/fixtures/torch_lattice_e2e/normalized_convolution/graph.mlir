module attributes {
  lattice.ir_version = 1,
  lattice.schema_digest = "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa",
  lattice.input_names = ["x_coords", "x_features", "x_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %x_coords: tensor<?x4xi32>,
    %x_features: tensor<?x2xf32>,
    %x_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %x = lattice.sparse.make %x_coords, %x_features, %x_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x2xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %conv_weight = lattice.weight @conv_weight {storage_key = "conv.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %conv_bias = lattice.weight @conv_bias {storage_key = "conv.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %conv = lattice.normalized_subm_conv3d %x, %conv_weight, %conv_bias {kernel_size = array<i64: 3, 1, 1>, dilation = array<i64: 1, 1, 1>, eps = 0.00000001 : f32} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %conv : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

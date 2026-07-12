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
    %x_features: tensor<?x3xf32>,
    %x_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %x = lattice.sparse.make %x_coords, %x_features, %x_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x3xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %encoder_weight = lattice.weight @encoder_weight {storage_key = "encoder.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %encoder_bias = lattice.weight @encoder_bias {storage_key = "encoder.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %encoder = lattice.normalized_subm_conv3d %x, %encoder_weight, %encoder_bias {kernel_size = array<i64: 3, 1, 1>, dilation = array<i64: 1, 1, 1>, eps = 0.00000001 : f32} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %down = lattice.pool3d %encoder {mode = #lattice.pool_mode<avg>, kernel_size = array<i64: 2, 1, 1>, stride = array<i64: 2, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %context_weight = lattice.weight @context_weight {storage_key = "context.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %context_bias = lattice.weight @context_bias {storage_key = "context.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %context = lattice.normalized_subm_conv3d %down, %context_weight, %context_bias {kernel_size = array<i64: 3, 1, 1>, dilation = array<i64: 1, 1, 1>, eps = 0.00000001 : f32} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %learned_up_weight = lattice.weight @learned_up_weight {storage_key = "learned_up.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %learned_up_bias = lattice.weight @learned_up_bias {storage_key = "learned_up.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %learned_up = lattice.target_normalized_conv_transpose3d %context, %x, %learned_up_weight, %learned_up_bias {kernel_size = array<i64: 3, 1, 1>, stride = array<i64: 2, 1, 1>, padding = array<i64: 1, 0, 0>, dilation = array<i64: 1, 1, 1>, eps = 0.00000001 : f32} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %reindex = lattice.sparse.reindex %learned_up, %x {fill = 0.0 : f32} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %pool_up = lattice.pool_transpose3d %context, %x {kernel_size = array<i64: 3, 1, 1>, stride = array<i64: 2, 1, 1>, padding = array<i64: 1, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %linear_up = lattice.trilinear_upsample3d %context, %x {stride = array<i64: 2, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %cat_1 = lattice.sparse.cat %reindex, %pool_up {join = #lattice.join<inner>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %cat_2 = lattice.sparse.cat %cat_1, %linear_up {join = #lattice.join<inner>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %output_weight = lattice.weight @output_weight {storage_key = "output.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %output_bias = lattice.weight @output_bias {storage_key = "output.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %output = lattice.normalized_subm_conv3d %cat_2, %output_weight, %output_bias {kernel_size = array<i64: 3, 1, 1>, dilation = array<i64: 1, 1, 1>, eps = 0.00000001 : f32} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %output : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

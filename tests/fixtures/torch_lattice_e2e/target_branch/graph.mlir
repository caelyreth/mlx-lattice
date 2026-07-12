module attributes {
  lattice.ir_version = 1,
  lattice.schema_digest = "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa",
  lattice.input_names = ["x_coords", "x_features", "x_active", "target_coords", "target_features", "target_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active", "sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %x_coords: tensor<?x4xi32>,
    %x_features: tensor<?x2xf32>,
    %x_active: tensor<1xi32>,
    %target_coords: tensor<?x4xi32>,
    %target_features: tensor<?x1xf32>,
    %target_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %x = lattice.sparse.make %x_coords, %x_features, %x_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x2xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %target = lattice.sparse.make %target_coords, %target_features, %target_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x1xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %left_weight = lattice.weight @left_weight {storage_key = "left.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %left_bias = lattice.weight @left_bias {storage_key = "left.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %left = lattice.conv3d %x, %left_weight, %left_bias {kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %right_weight = lattice.weight @right_weight {storage_key = "right.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %right = lattice.conv3d %x, %right_weight {kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %add = lattice.sparse.binary %left, %right {op = #lattice.binary_op<add>, join = #lattice.join<outer>, lhs_fill = 0.0 : f32, rhs_fill = 0.0 : f32} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %target_conv_weight = lattice.weight @target_conv_weight {storage_key = "target_conv.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %target_conv_bias = lattice.weight @target_conv_bias {storage_key = "target_conv.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %target_conv = lattice.target_conv3d %add, %target, %target_conv_weight, %target_conv_bias {kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %cat_1 = lattice.sparse.cat %add, %target_conv {join = #lattice.join<inner>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %cat_1 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

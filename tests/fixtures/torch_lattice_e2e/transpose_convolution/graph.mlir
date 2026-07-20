module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
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
    %down_weight = lattice.weight @down_weight {storage_key = "down.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %down = lattice.conv3d %x, %down_weight {kernel_size = array<i64: 2, 1, 1>, stride = array<i64: 2, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>, accumulation = "canonical_f32"} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %up_weight = lattice.weight @up_weight {storage_key = "up.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %up_bias = lattice.weight @up_bias {storage_key = "up.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %up = lattice.conv_transpose3d %down, %up_weight, %up_bias {kernel_size = array<i64: 2, 1, 1>, stride = array<i64: 2, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>, accumulation = "canonical_f32"} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %up : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

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
    %x = lattice.sparse.make %x_coords, %x_features, %x_active {stride = array<i64: 2, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x2xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %up_weight = lattice.weight @up_weight {storage_key = "up.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %up_bias = lattice.weight @up_bias {storage_key = "up.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %up = lattice.generative_conv_transpose3d %x, %up_weight, %up_bias {kernel_size = array<i64: 2, 1, 1>, stride = array<i64: 2, 1, 1>, accumulation = "canonical_f32"} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose, %act_features_in, %act_active = lattice.sparse.decompose %up : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x3xf32>, tensor<1xi32>)
    %act_features = lattice.activation %act_features_in {kind = #lattice.activation<tanh>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x3xf32>) -> tensor<?x3xf32>
    %act = lattice.sparse.with_features %up, %act_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x3xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %act : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

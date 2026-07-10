module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "545fd43029e2fbebc18404babc0ec463730a13056a18f51280d8f1a272a3369c",
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
    %input = lattice.sparse.make %coords, %features, %active {stride = array<i64: 2, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x?xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %up_weight = lattice.weight @up_weight {storage_key = "up.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %up_bias = lattice.weight @up_bias {storage_key = "up.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %up = lattice.generative_conv_transpose3d %input, %up_weight, %up_bias {kernel_size = array<i64: 2, 1, 1>, stride = array<i64: 2, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose, %act_features_in, %act_active = lattice.sparse.decompose %up : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x3xf32>, tensor<1xi32>)
    %act_features = lattice.activation %act_features_in {kind = #lattice.activation<tanh>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x3xf32>) -> tensor<?x3xf32>
    %act = lattice.sparse.with_features %up, %act_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x3xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %act : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

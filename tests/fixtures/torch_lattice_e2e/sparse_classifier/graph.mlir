module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "314833e397548364385e5a24c1faf5ebcd4eadc3a0d750a0bed444e2c855c4a1",
  lattice.input_names = ["coords", "features", "active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %coords: tensor<?x4xi32>,
    %features: tensor<?x?xf32>,
    %active: tensor<1xi32>
  ) -> tensor<?x2xf32> {
    %input = lattice.sparse.make %coords, %features, %active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x?xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stem_weight = lattice.weight @stem_weight {storage_key = "stem.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %stem_bias = lattice.weight @stem_bias {storage_key = "stem.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stem = lattice.conv3d %input, %stem_weight, %stem_bias {kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose, %norm_features_in, %norm_active = lattice.sparse.decompose %stem : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x4xf32>, tensor<1xi32>)
    %norm_weight = lattice.weight @norm_weight {storage_key = "norm.weight", layout = #lattice.weight_layout<channel_c>, packing = #lattice.packing<dense>} : !lattice.weight<channel, f32>
    %norm_bias = lattice.weight @norm_bias {storage_key = "norm.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %norm_running_mean = lattice.weight @norm_running_mean {storage_key = "norm.running_mean", layout = #lattice.weight_layout<channel_c>, packing = #lattice.packing<dense>} : !lattice.weight<channel, f32>
    %norm_running_var = lattice.weight @norm_running_var {storage_key = "norm.running_var", layout = #lattice.weight_layout<channel_c>, packing = #lattice.packing<dense>} : !lattice.weight<channel, f32>
    %norm_features = lattice.batch_norm %norm_features_in, %norm_weight, %norm_bias, %norm_running_mean, %norm_running_var {eps = 0.00001 : f32} : (tensor<?x4xf32>, !lattice.weight<channel, f32>, !lattice.weight<bias, f32>, !lattice.weight<channel, f32>, !lattice.weight<channel, f32>) -> tensor<?x4xf32>
    %norm = lattice.sparse.with_features %stem, %norm_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x4xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose1, %act_features_in, %act_active = lattice.sparse.decompose %norm : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x4xf32>, tensor<1xi32>)
    %act_features = lattice.activation %act_features_in {kind = #lattice.activation<relu>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %act = lattice.sparse.with_features %norm, %act_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x4xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %pool = lattice.pool3d %act {mode = #lattice.pool_mode<avg>, kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %global_pool = lattice.global_pool %pool {mode = #lattice.pool_mode<avg>, batch_size = 2} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> tensor<?x4xf32>
    %head_weight = lattice.weight @head_weight {storage_key = "head.weight", layout = #lattice.weight_layout<linear_o_i>, packing = #lattice.packing<dense>} : !lattice.weight<linear, f32>
    %head_bias = lattice.weight @head_bias {storage_key = "head.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %head = lattice.linear %global_pool, %head_weight, %head_bias  : (tensor<?x4xf32>, !lattice.weight<linear, f32>, !lattice.weight<bias, f32>) -> tensor<?x2xf32>
    return %head : tensor<?x2xf32>
  }
}

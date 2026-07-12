module attributes {
  lattice.ir_version = 1,
  lattice.schema_digest = "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa",
  lattice.input_names = ["x_coords", "x_features", "x_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %x_coords: tensor<?x4xi32>,
    %x_features: tensor<?x3xf32>,
    %x_active: tensor<1xi32>
  ) -> tensor<?x2xf32> {
    %x = lattice.sparse.make %x_coords, %x_features, %x_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x3xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stem_weight = lattice.weight @stem_weight {storage_key = "stem.weight", layout = #lattice.weight_layout<conv3d_o_xyz_i>, packing = #lattice.packing<int8, group_size = 32, scale_dtype = f16, mode = affine>} : !lattice.weight<conv3d, f32>
    %stem_bias = lattice.weight @stem_bias {storage_key = "stem.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stem = lattice.conv3d %x, %stem_weight, %stem_bias {kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose, %act_features_in, %act_active = lattice.sparse.decompose %stem : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x4xf32>, tensor<1xi32>)
    %act_features = lattice.activation %act_features_in {kind = #lattice.activation<silu>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %act = lattice.sparse.with_features %stem, %act_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x4xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %global_pool = lattice.global_pool %act {mode = #lattice.pool_mode<avg>, batch_size = 2} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> tensor<?x4xf32>
    %head_weight = lattice.weight @head_weight {storage_key = "head.weight", layout = #lattice.weight_layout<linear_o_i>, packing = #lattice.packing<int8, group_size = 32, scale_dtype = f16, mode = affine>} : !lattice.weight<linear, f32>
    %head_bias = lattice.weight @head_bias {storage_key = "head.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %head = lattice.linear %global_pool, %head_weight, %head_bias  : (tensor<?x4xf32>, !lattice.weight<linear, f32>, !lattice.weight<bias, f32>) -> tensor<?x2xf32>
    return %head : tensor<?x2xf32>
  }
}

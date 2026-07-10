module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "8a5ace10e29b47304594c1b66608ab64318c68568a69f4dcbc1ed8c570d73088",
  lattice.input_names = ["lhs_coords", "lhs_features", "lhs_active", "rhs_coords", "rhs_features", "rhs_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active", "sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %lhs_coords: tensor<?x4xi32>,
    %lhs_features: tensor<?x8xf16>,
    %lhs_active: tensor<1xi32>,
    %rhs_coords: tensor<?x4xi32>,
    %rhs_features: tensor<?x16xf16>,
    %rhs_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f16> {
    %lhs = lattice.sparse.make %lhs_coords, %lhs_features, %lhs_active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x8xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f16>
    %rhs = lattice.sparse.make %rhs_coords, %rhs_features, %rhs_active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x16xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f16>
    %out = lattice.sparse.cat %lhs, %rhs
      {join = #lattice.join<outer>}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f16>,
         !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f16>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f16>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f16>
  }
}

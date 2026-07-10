// Invalid: sparse binary op must be one of the supported elementwise modes.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "33a97d62e5b150b98940c62284f42b326e879cc4aca2747cdbc0d77c851f66c7",
  lattice.input_names = ["input0", "input1", "input2", "input3", "input4", "input5"],
  lattice.input_roles = ["tensor", "tensor", "tensor", "tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %lhs_coords: tensor<?x4xi32>,
    %lhs_features: tensor<?x32xf16>,
    %lhs_active: tensor<1xi32>,
    %rhs_coords: tensor<?x4xi32>,
    %rhs_features: tensor<?x32xf16>,
    %rhs_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                              feature = row_channel, dtype = f16> {
    %lhs = lattice.sparse.make %lhs_coords, %lhs_features, %lhs_active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    %rhs = lattice.sparse.make %rhs_coords, %rhs_features, %rhs_active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    %out = lattice.sparse.binary %lhs, %rhs
      {op = #lattice.binary_op<divide>, join = #lattice.join<inner>,
       lhs_fill = 0.0 : f32, rhs_fill = 0.0 : f32}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f16>,
         !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f16>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                        feature = row_channel, dtype = f16>
  }
}

// Invalid: normalized convolution epsilon must be positive.
module attributes {
  lattice.ir_version = 1,
  lattice.schema_digest = "eb5aaff9fc917038f49f4c62f9e19c2d78d2b3540035de55c270b9513d3156aa",
  lattice.input_names = ["coords", "features", "active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %coords: tensor<?x4xi32>,
    %features: tensor<?x32xf16>,
    %active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                              feature = row_channel, dtype = f16> {
    %input = lattice.sparse.make %coords, %features, %active
      {stride = array<i64: 1, 1, 1>,
       coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    %weight = lattice.weight @block.weight
      {storage_key = "block.weight",
       layout = #lattice.weight_layout<conv3d_o_xyz_i>,
       packing = #lattice.packing<dense>}
      : !lattice.weight<conv3d, f16>
    %out = lattice.normalized_subm_conv3d %input, %weight
      {kernel_size = array<i64: 3, 3, 3>,
       dilation = array<i64: 1, 1, 1>, eps = 0.0 : f32}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f16>,
         !lattice.weight<conv3d, f16>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                         feature = row_channel, dtype = f16>
  }
}

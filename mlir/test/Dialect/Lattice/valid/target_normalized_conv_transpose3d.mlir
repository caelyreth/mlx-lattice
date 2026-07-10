// Valid: normalized transpose convolution may consume explicit target support.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "e48cb610f907d8c7afbe66c197f2e01ab7ba3519a3f3d452b9643768f5c476c9",
  lattice.input_names = ["source_coords", "source_features", "source_active", "target_coords", "target_features", "target_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active", "sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%source_coords: tensor<?x4xi32>,
                     %source_features: tensor<?x1xf32>,
                     %source_active: tensor<1xi32>,
                     %target_coords: tensor<?x4xi32>,
                     %target_features: tensor<?x1xf32>,
                     %target_active: tensor<1xi32>)
      -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f32> {
    %source = lattice.sparse.make %source_coords, %source_features, %source_active
      {stride = array<i64: 2, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x1xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    %target = lattice.sparse.make %target_coords, %target_features, %target_active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x1xf32>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f32>
    %weight = lattice.weight @up.weight
      {storage_key = "up.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>,
       packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %out = lattice.target_normalized_conv_transpose3d %source, %target, %weight
      {kernel_size = array<i64: 3, 1, 1>, stride = array<i64: 2, 1, 1>,
       padding = array<i64: 1, 0, 0>, dilation = array<i64: 1, 1, 1>,
       eps = 1.0e-8 : f32}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>,
         !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>,
         !lattice.weight<conv3d, f32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                        feature = row_channel, dtype = f32>
  }
}

// Invalid: quantized conv must be lattice.conv3d with quantized weight type.
module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "545fd43029e2fbebc18404babc0ec463730a13056a18f51280d8f1a272a3369c",
  lattice.input_names = ["input0", "input1", "input2"],
  lattice.input_roles = ["tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(%coords: tensor<?x4xi32>,
                     %features: tensor<?x32xf16>,
                     %active: tensor<1xi32>)
      -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f16> {
    %input = lattice.sparse.make %coords, %features, %active
      {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    %weight = lattice.weight @stem.qweight
      {storage_key = "stem.qweight",
       layout = #lattice.weight_layout<conv3d_o_zyx_i>,
       packing = #lattice.packing<int4, group_size = 32,
                                  scale_dtype = f16, mode = affine>}
      : !lattice.weight<conv3d, i4>
    %out = lattice.quantized_conv3d %input, %weight
      {kernel_size = array<i64: 3, 3, 3>, stride = array<i64: 1, 1, 1>,
       padding = array<i64: 1, 1, 1>, dilation = array<i64: 1, 1, 1>}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f16>,
         !lattice.weight<conv3d, i4>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                        feature = row_channel, dtype = f16>
  }
}

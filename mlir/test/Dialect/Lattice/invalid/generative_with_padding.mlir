// Invalid: generative transpose convolution must not carry padding/dilation.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
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
      {stride = array<i64: 2, 2, 2>, coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    %weight = lattice.weight @decoder.gen.weight
      {storage_key = "decoder.gen.weight",
       layout = #lattice.weight_layout<conv3d_o_xyz_i>,
       packing = #lattice.packing<dense>}
      : !lattice.weight<conv3d, f16>
    %out = lattice.generative_conv_transpose3d %input, %weight
      {kernel_size = array<i64: 2, 2, 2>,
       stride = array<i64: 2, 2, 2>,
       padding = array<i64: 0, 0, 0>}
      : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                feature = row_channel, dtype = f16>,
         !lattice.weight<conv3d, f16>)
        -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                  feature = row_channel, dtype = f16>
    return %out : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z,
                                        feature = row_channel, dtype = f16>
  }
}

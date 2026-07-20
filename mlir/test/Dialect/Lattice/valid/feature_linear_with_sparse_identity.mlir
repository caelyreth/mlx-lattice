// Valid: feature-only linear update preserves sparse identity explicitly.
module attributes {
  lattice.ir_version = 2,
  lattice.schema_digest = "1380f1e819fc0eb1af587202ecec3c14ec2c981d249333c5061f0263f82072ad",
  lattice.input_names = ["input0", "input1", "input2"],
  lattice.input_roles = ["tensor", "tensor", "tensor"],
  lattice.output_names = ["output0"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %coords: tensor<?x4xi32>,
    %features: tensor<?x32xf16>,
    %active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3,
                              coord = batch_x_y_z,
                              feature = row_channel,
                              dtype = f16> {
    %input = lattice.sparse.make %coords, %features, %active
      {stride = array<i64: 1, 1, 1>,
       coord_order = #lattice.coord<batch_x_y_z>}
      : (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)
        -> !lattice.sparse_tensor<rank = 3,
                                  coord = batch_x_y_z,
                                  feature = row_channel,
                                  dtype = f16>

    %coords0, %features0, %active0 = lattice.sparse.decompose %input
      : !lattice.sparse_tensor<rank = 3,
                               coord = batch_x_y_z,
                               feature = row_channel,
                               dtype = f16>
        -> (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)

    %weight = lattice.weight @mlp.proj.weight
      {storage_key = "mlp.proj.weight",
       layout = #lattice.weight_layout<linear_o_i>,
       packing = #lattice.packing<dense>}
      : !lattice.weight<linear, f16>

    %bias = lattice.weight @mlp.proj.bias
      {storage_key = "mlp.proj.bias",
       layout = #lattice.weight_layout<bias_c>,
       packing = #lattice.packing<dense>}
      : !lattice.weight<bias, f16>

    %projected = lattice.linear %features0, %weight, %bias
      : (tensor<?x32xf16>,
         !lattice.weight<linear, f16>,
         !lattice.weight<bias, f16>)
        -> tensor<?x64xf16>

    %out = lattice.sparse.with_features %input, %projected
      : (!lattice.sparse_tensor<rank = 3,
                                coord = batch_x_y_z,
                                feature = row_channel,
                                dtype = f16>,
         tensor<?x64xf16>)
        -> !lattice.sparse_tensor<rank = 3,
                                  coord = batch_x_y_z,
                                  feature = row_channel,
                                  dtype = f16>

    return %out : !lattice.sparse_tensor<rank = 3,
                                        coord = batch_x_y_z,
                                        feature = row_channel,
                                        dtype = f16>
  }
}

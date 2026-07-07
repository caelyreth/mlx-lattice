// Valid: dense normalization updates feature rows while preserving sparse support explicitly.
module attributes {
  lattice.ir_version = 0,
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

    %scale = lattice.weight @norm.scale
      {storage_key = "norm.scale",
       layout = #lattice.weight_layout<channel_c>,
       packing = #lattice.packing<dense>}
      : !lattice.weight<channel, f16>

    %bias = lattice.weight @norm.bias
      {storage_key = "norm.bias",
       layout = #lattice.weight_layout<bias_c>,
       packing = #lattice.packing<dense>}
      : !lattice.weight<bias, f16>

    %mean = lattice.weight @norm.mean
      {storage_key = "norm.mean",
       layout = #lattice.weight_layout<channel_c>,
       packing = #lattice.packing<dense>}
      : !lattice.weight<channel, f16>

    %var = lattice.weight @norm.var
      {storage_key = "norm.var",
       layout = #lattice.weight_layout<channel_c>,
       packing = #lattice.packing<dense>}
      : !lattice.weight<channel, f16>

    %bn = lattice.batch_norm %features0, %scale, %bias, %mean, %var
      {eps = 1.0e-05 : f32}
      : (tensor<?x32xf16>,
         !lattice.weight<channel, f16>,
         !lattice.weight<bias, f16>,
         !lattice.weight<channel, f16>,
         !lattice.weight<channel, f16>)
        -> tensor<?x32xf16>

    %ln = lattice.layer_norm %bn, %scale, %bias
      {eps = 1.0e-05 : f32}
      : (tensor<?x32xf16>,
         !lattice.weight<channel, f16>,
         !lattice.weight<bias, f16>)
        -> tensor<?x32xf16>

    %rms = lattice.rms_norm %ln, %scale
      {eps = 1.0e-05 : f32}
      : (tensor<?x32xf16>, !lattice.weight<channel, f16>)
        -> tensor<?x32xf16>

    %out = lattice.sparse.with_features %input, %rms
      : (!lattice.sparse_tensor<rank = 3,
                                coord = batch_x_y_z,
                                feature = row_channel,
                                dtype = f16>,
         tensor<?x32xf16>)
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

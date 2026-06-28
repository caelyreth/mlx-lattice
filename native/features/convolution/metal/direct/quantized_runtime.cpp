#include "features/convolution/metal/direct/quantized_runtime.h"

#include "platform/metal/runtime_utils.h"

namespace mlx_lattice::backend::metal::conv::quantized::direct {

void encode(
    QuantizedSparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
#ifdef _METAL_
    auto library = lattice_library(stream);
    auto& encoder = command_encoder(stream);
    auto fp16 = inputs[0].dtype() == mx::float16;
    auto kernel = lattice_kernel(
        stream,
        fp16 ? (shape.bits == 4 ? "sparse_quantized_conv_f16_i32_b4"
                                : "sparse_quantized_conv_f16_i32_b8")
             : (shape.bits == 4 ? "sparse_quantized_conv_f32_i32_b4"
                                : "sparse_quantized_conv_f32_i32_b8"),
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    bind_input_arrays(encoder, inputs);
    encoder.set_output_array(out, 9);
    set_bytes_range(
        encoder,
        10,
        static_cast<int>(inputs[4].shape(0)),
        shape.out_capacity,
        shape.in_channels,
        shape.out_channels,
        shape.storage_in_channels,
        shape.group_size,
        stride_i32(inputs[0], 0),
        stride_i32(inputs[0], 1)
    );
    constexpr int kOutputTile = 8;
    auto channel_blocks = (shape.out_channels + kOutputTile - 1) / kOutputTile;
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(shape.out_capacity) * channel_blocks
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::conv::quantized::direct

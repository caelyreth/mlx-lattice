#include "features/convolution/metal/tensor_ops/input_grad/runtime.h"

#include <stdexcept>

#include "platform/metal/capabilities.h"
#include "platform/metal/runtime_utils.h"

namespace mlx_lattice::backend::metal::tensor_ops::conv::input_grad {
namespace {

constexpr int kChannels = 16;
constexpr int kMinInputRows = 32768;

} // namespace

bool supports(SparseConvShape shape) {
    return shape.in_channels == kChannels && shape.out_channels == kChannels &&
           shape.n_kernels >= 16 && shape.weight_layout == 0;
}

bool is_preferred(SparseConvShape shape, const mx::Stream& stream) {
    return supports(shape) && shape.in_capacity >= kMinInputRows &&
           has_neural_acceleration(stream);
}

void encode(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
#ifdef _METAL_
    auto library = lattice_library(stream);
    auto& encoder = command_encoder(stream);

    auto kernel = lattice_kernel(
        stream, "sparse_relation_conv_input_grad_tensor_ops_f32_i32", library
    );
    encoder.set_compute_pipeline_state(kernel);
    bind_input_arrays(encoder, inputs, {0, 1, 3, 4, 5, 7, 8});
    encoder.set_output_array(out, 7);
    set_bytes_range(
        encoder,
        8,
        static_cast<int>(inputs[2].shape(0)),
        shape.out_capacity,
        shape.in_capacity,
        shape.n_kernels,
        stride_i32(inputs[0], 0),
        stride_i32(inputs[0], 1),
        stride_i32(inputs[1], 0),
        stride_i32(inputs[1], 1),
        stride_i32(inputs[1], 2)
    );
    auto total_tiles =
        static_cast<size_t>((shape.in_capacity + kChannels - 1) / kChannels);
    encoder.dispatch_threadgroups(
        MTL::Size(total_tiles, 1, 1), MTL::Size(32, 1, 1)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::input_grad

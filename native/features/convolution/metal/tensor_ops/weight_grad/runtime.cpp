#include "features/convolution/metal/tensor_ops/weight_grad/runtime.h"

#include <cstddef>
#include <stdexcept>

#include "platform/metal/capabilities.h"
#include "platform/metal/runtime_utils.h"

namespace mlx_lattice::backend::metal::tensor_ops::conv::weight_grad {
namespace {

constexpr int kChannels = 16;
constexpr int kPartitionEdges = 2048;
constexpr int kMaxPartitions = 64;
constexpr int kMinInputRows = 32768;

#ifdef _METAL_
bool is_float16(const mx::array& array) { return array.dtype() == mx::float16; }

const char* contract_kernel_name(bool fp16) {
    return fp16 ? "sparse_relation_conv_weight_grad_tensor_ops_f16_i32"
                : "sparse_relation_conv_weight_grad_tensor_ops_f32_i32";
}

const char* reduce_kernel_name(bool fp16) {
    return fp16 ? "sparse_relation_conv_weight_grad_tensor_ops_reduce_f16"
                : "sparse_relation_conv_weight_grad_tensor_ops_reduce_f32";
}

int partition_count(SparseConvShape shape) {
    auto partitions =
        (shape.in_capacity + kPartitionEdges - 1) / kPartitionEdges;
    return std::clamp(partitions, 1, kMaxPartitions);
}

#endif

} // namespace

bool supports(SparseConvShape shape) {
    auto supported_channels = [](int channels) {
        return channels == 16 || channels == 32 || channels == 64;
    };
    return supported_channels(shape.in_channels) &&
           supported_channels(shape.out_channels) && shape.n_kernels >= 16;
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
    auto partitions = partition_count(shape);
    auto in_channel_blocks = shape.in_channels / kChannels;
    auto out_channel_blocks = shape.out_channels / kChannels;
    auto channel_tiles = in_channel_blocks * out_channel_blocks;
    auto partial_values = static_cast<std::size_t>(partitions) *
                          static_cast<std::size_t>(shape.n_kernels) *
                          static_cast<std::size_t>(channel_tiles) * kChannels *
                          kChannels;
    auto partials = make_temp<float>(partial_values);

    auto library = lattice_library(stream);
    auto& encoder = command_encoder(stream);
    encoder.add_temporary(partials);

    auto fp16 = is_float16(inputs[0]);
    auto contract = lattice_kernel(stream, contract_kernel_name(fp16), library);
    encoder.set_compute_pipeline_state(contract);
    bind_input_arrays(encoder, inputs, {0, 1, 2, 3, 5, 7, 8});
    encoder.set_output_array(partials, 7);
    set_bytes_range(
        encoder,
        8,
        static_cast<int>(inputs[2].shape(0)),
        shape.out_capacity,
        shape.n_kernels,
        partitions,
        stride_i32(inputs[0], 0),
        stride_i32(inputs[0], 1),
        stride_i32(inputs[1], 0),
        stride_i32(inputs[1], 1),
        shape.in_channels,
        shape.out_channels
    );
    encoder.dispatch_threadgroups(
        MTL::Size(
            static_cast<std::size_t>(shape.n_kernels) *
                static_cast<std::size_t>(partitions) *
                static_cast<std::size_t>(channel_tiles),
            1,
            1
        ),
        MTL::Size(32, 1, 1)
    );

    auto reduce = lattice_kernel(stream, reduce_kernel_name(fp16), library);
    encoder.set_compute_pipeline_state(reduce);
    encoder.set_input_array(partials, 0);
    encoder.set_output_array(out, 1);
    set_bytes_range(
        encoder,
        2,
        shape.n_kernels,
        partitions,
        shape.weight_layout,
        shape.kernel_x,
        shape.kernel_y,
        shape.kernel_z,
        shape.in_channels,
        shape.out_channels
    );
    dispatch_1d(
        encoder,
        reduce,
        static_cast<std::size_t>(shape.n_kernels) *
            static_cast<std::size_t>(channel_tiles) * kChannels * kChannels
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::weight_grad

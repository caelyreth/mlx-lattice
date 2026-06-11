#include "backends/metal/tensor_ops/conv/forward/runtime.h"

#include <algorithm>
#include <stdexcept>

#include "backends/metal/runtime_utils.h"
#include "backends/metal/tensor_ops/capabilities.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::tensor_ops::conv::forward {
namespace {

constexpr int kChannels = 16;
constexpr int kTileEdges = 16;
constexpr int kMinInputRows = 32768;
constexpr bool kEnableExperimentalForward = false;

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}

#ifdef _METAL_
template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, size_t elements) {
    auto threads = std::max<size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}

template <typename Encoder, typename Library>
void clear_output(
    Encoder& encoder,
    mx::metal::Device& device,
    Library library,
    mx::array& out
) {
    auto clear = device.get_kernel("sparse_relation_conv_clear_f32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(out, 0);
    auto total = static_cast<int>(out.size());
    encoder.set_bytes(total, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(total));
}
#endif

} // namespace

bool supports(SparseConvShape shape) {
    return shape.in_channels == kChannels && shape.out_channels == kChannels &&
           shape.n_kernels >= 16 && shape.weight_layout == 0;
}

bool is_preferred(SparseConvShape shape, const mx::Stream& stream) {
    // Experimental route: kernel-grouped implicit GEMM currently pays atomic
    // scatter overhead after the TensorOps tile, so keep it disabled until the
    // planner can emit output-owned or segmented-reduction tiles.
    return kEnableExperimentalForward && supports(shape) &&
           shape.in_capacity >= kMinInputRows && has_nax_acceleration(stream);
}

void encode(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
#ifdef _METAL_
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    clear_output(encoder, device, library, out);

    auto kernel = device.get_kernel(
        "sparse_relation_conv_forward_tensor_ops_f32_i32", library
    );
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(inputs[3], 3);
    encoder.set_input_array(inputs[5], 4);
    encoder.set_input_array(inputs[9], 5);
    encoder.set_input_array(inputs[10], 6);
    encoder.set_output_array(out, 7);
    encoder.set_bytes(static_cast<int>(inputs[2].shape(0)), 8);
    encoder.set_bytes(shape.out_capacity, 9);
    encoder.set_bytes(shape.n_kernels, 10);
    encoder.set_bytes(stride_at(inputs[0], 0), 11);
    encoder.set_bytes(stride_at(inputs[0], 1), 12);
    encoder.set_bytes(stride_at(inputs[1], 0), 13);
    encoder.set_bytes(stride_at(inputs[1], 1), 14);
    encoder.set_bytes(stride_at(inputs[1], 2), 15);
    auto total_tiles =
        static_cast<size_t>(shape.n_kernels) *
        static_cast<size_t>(
            (static_cast<int>(inputs[2].shape(0)) + kTileEdges - 1) / kTileEdges
        );
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

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::forward

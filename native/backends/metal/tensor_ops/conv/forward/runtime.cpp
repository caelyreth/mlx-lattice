#include "backends/metal/tensor_ops/conv/forward/runtime.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "backends/metal/tensor_ops/capabilities.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::tensor_ops::conv::forward {
namespace {

constexpr int kChannels = 16;
constexpr int kMinInputRows = 100000;

#ifdef _METAL_
template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, std::size_t elements) {
    auto threads = std::max<std::size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}

bool is_float16(const mx::array& array) { return array.dtype() == mx::float16; }

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}
#endif

} // namespace

bool supports(SparseConvShape shape) {
    return shape.in_channels == 64 && shape.out_channels == 64 &&
           shape.n_kernels == 27;
}

bool is_preferred(SparseConvShape shape, const mx::Stream& stream) {
    return supports(shape) && shape.in_capacity >= kMinInputRows &&
           has_nax_acceleration(stream);
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
    auto kernel = device.get_kernel(
        is_float16(inputs[0])
            ? "sparse_relation_conv_forward_implicit_gemm_f16_i32_c64"
            : "sparse_relation_conv_forward_implicit_gemm_f32_i32_c64",
        library
    );

    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 7);
    encoder.set_bytes(static_cast<int>(inputs[2].shape(0)), 8);
    encoder.set_bytes(shape.out_capacity, 9);
    encoder.set_bytes(shape.n_kernels, 10);
    encoder.set_bytes(shape.in_channels, 11);
    encoder.set_bytes(shape.out_channels, 12);
    encoder.set_bytes(stride_at(inputs[0], 0), 13);
    encoder.set_bytes(stride_at(inputs[0], 1), 14);
    encoder.set_bytes(stride_at(inputs[1], 0), 15);
    encoder.set_bytes(stride_at(inputs[1], 1), 16);
    encoder.set_bytes(stride_at(inputs[1], 2), 17);
    encoder.set_bytes(inputs[1].ndim() == 5 ? stride_at(inputs[1], 3) : 0, 18);
    encoder.set_bytes(inputs[1].ndim() == 5 ? stride_at(inputs[1], 4) : 0, 19);
    encoder.set_bytes(shape.weight_layout, 20);
    encoder.set_bytes(shape.kernel_x, 21);
    encoder.set_bytes(shape.kernel_y, 22);
    encoder.set_bytes(shape.kernel_z, 23);
    auto row_tiles = static_cast<std::size_t>(
        (shape.out_capacity + kChannels - 1) / kChannels
    );
    auto co_blocks = static_cast<std::size_t>(shape.out_channels / kChannels);
    dispatch_1d(encoder, kernel, row_tiles * co_blocks);
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::forward

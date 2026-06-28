#include "features/convolution/metal/sorted_igemm/quantized_runtime.h"

#include <cstddef>
#include <stdexcept>

#include "platform/metal/capabilities.h"
#include "platform/metal/runtime_utils.h"

namespace mlx_lattice::backend::metal::conv::quantized::sorted_igemm {
namespace {

constexpr int kTileRows = 64;
constexpr int kTileChannels = 32;

#ifdef _METAL_
const char* kernel_name(QuantizedSparseConvShape shape) {
    if (shape.in_channels == 32 && shape.out_channels == 32) {
        return shape.bits == 4 ? "sparse_quantized_igemm_f16_b4_cin32_cout32"
                               : "sparse_quantized_igemm_f16_b8_cin32_cout32";
    }
    if (shape.in_channels == 32 && shape.out_channels == 64) {
        return shape.bits == 4 ? "sparse_quantized_igemm_f16_b4_cin32_cout64"
                               : "sparse_quantized_igemm_f16_b8_cin32_cout64";
    }
    if (shape.in_channels == 64 && shape.out_channels == 32) {
        return shape.bits == 4 ? "sparse_quantized_igemm_f16_b4_cin64_cout32"
                               : "sparse_quantized_igemm_f16_b8_cin64_cout32";
    }
    if (shape.in_channels == 64 && shape.out_channels == 64) {
        return shape.bits == 4 ? "sparse_quantized_igemm_f16_b4_cin64_cout64"
                               : "sparse_quantized_igemm_f16_b8_cin64_cout64";
    }
    throw std::invalid_argument(
        "quantized implicit GEMM received an unsupported channel shape."
    );
}

#endif

} // namespace

bool is_preferred(
    QuantizedSparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs
) {
    auto supported_channels =
        (shape.in_channels == 32 || shape.in_channels == 64) &&
        (shape.out_channels == 32 || shape.out_channels == 64);
    return inputs.size() == 12 && shape.n_kernels == 27 &&
           shape.storage_in_channels == shape.in_channels &&
           shape.group_size <= shape.in_channels && supported_channels &&
           inputs[0].dtype() == mx::float16 &&
           tensor_ops::capability_tier(stream) !=
               tensor_ops::CapabilityTier::unavailable;
}

void encode(
    QuantizedSparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
#ifdef _METAL_
    if (!is_preferred(shape, stream, inputs)) {
        throw std::invalid_argument(
            "quantized implicit GEMM requires a sorted fp16 K=27 problem "
            "with C_in and C_out in {32, 64}."
        );
    }
    auto sorted = make_temp<mx::float16_t>(
        static_cast<std::size_t>(shape.out_capacity) * shape.out_channels
    );
    auto library = lattice_library(stream);
    auto& encoder = command_encoder(stream);

    encoder.add_temporary(sorted);

    auto contract = lattice_kernel(stream, kernel_name(shape), library);
    encoder.set_compute_pipeline_state(contract);
    bind_input_arrays(encoder, inputs, {0, 1, 2, 3, 9, 11});
    encoder.set_output_array(sorted, 6);
    set_bytes_range(encoder, 7, shape.out_capacity, shape.group_size);
    encoder.dispatch_threadgroups(
        MTL::Size(
            static_cast<size_t>(shape.out_channels / kTileChannels),
            static_cast<size_t>(
                (shape.out_capacity + kTileRows - 1) / kTileRows
            ),
            1
        ),
        MTL::Size(128, 1, 1)
    );

    auto reorder =
        lattice_kernel(stream, "sparse_quantized_igemm_reorder_f16", library);
    encoder.set_compute_pipeline_state(reorder);
    encoder.set_input_array(sorted, 0);
    encoder.set_input_array(inputs[10], 1);
    encoder.set_output_array(out, 2);
    set_bytes_range(encoder, 3, shape.out_capacity, shape.out_channels);
    dispatch_1d(
        encoder,
        reorder,
        static_cast<size_t>(shape.out_capacity) * shape.out_channels
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::conv::quantized::sorted_igemm

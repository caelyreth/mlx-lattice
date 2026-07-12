#include "features/convolution/metal/runtime.h"

#include <stdexcept>

#include "foundation/array_utils.h"
#include "platform/metal/capabilities.h"
#include "platform/metal/runtime_utils.h"

namespace mlx_lattice::backend::metal::conv {

void eval_projection(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    const auto rows = static_cast<int>(inputs[0].shape(0));
    const auto in_channels = static_cast<int>(inputs[0].shape(1));
    const auto out_channels = static_cast<int>(inputs[1].shape(0));
    auto library = lattice_library(stream);
    auto& encoder = command_encoder(stream);
    const auto use_tensor_ops = in_channels == 32 && out_channels == 32 &&
                                tensor_ops::has_neural_acceleration(stream);
    auto kernel = lattice_kernel(
        stream,
        use_tensor_ops ? "precise_feature_projection_f32_c32"
                       : "precise_feature_projection_f32",
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(out, 2);
    set_bytes_range(encoder, 3, rows, in_channels, out_channels);
    if (use_tensor_ops) {
        encoder.dispatch_threadgroups(
            MTL::Size(static_cast<size_t>((rows + 63) / 64), 1, 1),
            MTL::Size(128, 1, 1)
        );
    } else {
        dispatch_1d(
            encoder,
            kernel,
            static_cast<size_t>(rows) * static_cast<size_t>(out_channels)
        );
    }
#else
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::conv

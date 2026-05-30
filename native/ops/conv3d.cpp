#include "ops/conv3d.h"

#include <dlfcn.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx_lattice {

namespace {

// MARK: - helpers

std::string binary_dir() {
    static std::string dir = [] {
        Dl_info info;
        if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
            throw std::runtime_error("Unable to resolve native module path.");
        }
        return std::filesystem::path(info.dli_fname).parent_path().string();
    }();
    return dir;
}

void validate_conv3d_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int out_rows
) {
    if (out_rows < 0) {
        throw std::invalid_argument("out_rows must be non-negative.");
    }
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, Cin).");
    }
    if (weight.ndim() != 3) {
        throw std::invalid_argument("weight must have shape (K, Cin, Cout).");
    }
    if (maps.ndim() != 2 || maps.shape(1) != 2) {
        throw std::invalid_argument("maps must have shape (M, 2).");
    }
    if (kernels.ndim() != 1 || kernels.shape(0) != maps.shape(0)) {
        throw std::invalid_argument("kernels must have shape (M,).");
    }
    if (feats.dtype() != mx::float32 || weight.dtype() != mx::float32) {
        throw std::invalid_argument("conv3d_feats supports float32 tensors.");
    }
    if (maps.dtype() != mx::int32 || kernels.dtype() != mx::int32) {
        throw std::invalid_argument("maps and kernels must be int32.");
    }
    if (weight.shape(1) != feats.shape(1)) {
        throw std::invalid_argument("weight input channels must match feats.");
    }
}

class Conv3dFeats : public mx::Primitive {
  public:
    Conv3dFeats(mx::Stream stream, int rows, int in_channels, int out_channels)
        : mx::Primitive(stream), rows_(rows), in_channels_(in_channels),
          out_channels_(out_channels) {}

    // MARK: - primitive

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        const auto& feats = inputs[0];
        const auto& weight = inputs[1];
        const auto& maps = inputs[2];
        const auto& kernels = inputs[3];
        auto& out = outputs[0];

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& encoder = mx::cpu::get_command_encoder(stream());
        encoder.set_input_array(feats);
        encoder.set_input_array(weight);
        encoder.set_input_array(maps);
        encoder.set_input_array(kernels);
        encoder.set_output_array(out);

        encoder.dispatch([feats_ptr = feats.data<float>(),
                          weight_ptr = weight.data<float>(),
                          maps_ptr = maps.data<int32_t>(),
                          kernels_ptr = kernels.data<int32_t>(),
                          out_ptr = out.data<float>(),
                          pairs = maps.shape(0),
                          rows = rows_,
                          in_channels = in_channels_,
                          out_channels = out_channels_]() {
            std::fill(out_ptr, out_ptr + rows * out_channels, 0.0f);
            for (int pair = 0; pair < pairs; ++pair) {
                int in_row = maps_ptr[pair * 2];
                int out_row = maps_ptr[pair * 2 + 1];
                int kernel = kernels_ptr[pair];
                for (int out_col = 0; out_col < out_channels; ++out_col) {
                    float acc = 0.0f;
                    for (int in_col = 0; in_col < in_channels; ++in_col) {
                        acc += feats_ptr[in_row * in_channels + in_col] *
                               weight_ptr
                                   [(kernel * in_channels + in_col) *
                                        out_channels +
                                    out_col];
                    }
                    out_ptr[out_row * out_channels + out_col] += acc;
                }
            }
        });
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
#ifdef _METAL_
        const auto& feats = inputs[0];
        const auto& weight = inputs[1];
        const auto& maps = inputs[2];
        const auto& kernels = inputs[3];
        auto& out = outputs[0];

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& s = stream();
        auto& device = mx::metal::device(s.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(s);

        auto zero = device.get_kernel("fill_zero_float32", library);
        int out_size = rows_ * out_channels_;
        if (out_size > 0) {
            encoder.set_compute_pipeline_state(zero);
            encoder.set_output_array(out, 0);
            encoder.set_bytes(out_size, 1);
            auto zero_group = std::min(
                static_cast<size_t>(out_size),
                zero->maxTotalThreadsPerThreadgroup()
            );
            encoder.dispatch_threads(
                MTL::Size(static_cast<size_t>(out_size), 1, 1),
                MTL::Size(zero_group, 1, 1)
            );
        }

        auto conv = device.get_kernel("conv3d_feats_float32", library);
        auto elements = static_cast<size_t>(maps.shape(0)) *
                        static_cast<size_t>(out_channels_);
        if (elements == 0) {
            return;
        }
        encoder.set_compute_pipeline_state(conv);
        encoder.set_input_array(feats, 0);
        encoder.set_input_array(weight, 1);
        encoder.set_input_array(maps, 2);
        encoder.set_input_array(kernels, 3);
        encoder.set_output_array(out, 4);
        int pair_count = maps.shape(0);
        encoder.set_bytes(pair_count, 5);
        encoder.set_bytes(in_channels_, 6);
        encoder.set_bytes(out_channels_, 7);
        auto conv_group =
            std::min(elements, conv->maxTotalThreadsPerThreadgroup());
        encoder.dispatch_threads(
            MTL::Size(elements, 1, 1), MTL::Size(conv_group, 1, 1)
        );
#else
        throw std::runtime_error("Metal support is not available.");
#endif
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Conv3dFeats has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("Conv3dFeats has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("Conv3dFeats has no vmap implementation.");
    }

    const char* name() const override { return "Conv3dFeats"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& conv = static_cast<const Conv3dFeats&>(other);
        return rows_ == conv.rows_ && in_channels_ == conv.in_channels_ &&
               out_channels_ == conv.out_channels_;
    }

  private:
    int rows_;
    int in_channels_;
    int out_channels_;
};

} // namespace

// MARK: - api

mx::array conv3d_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int out_rows,
    mx::StreamOrDevice stream
) {
    validate_conv3d_feats(feats, weight, maps, kernels, out_rows);

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{out_rows, weight.shape(2)},
        mx::float32,
        std::make_shared<Conv3dFeats>(
            s, out_rows, feats.shape(1), weight.shape(2)
        ),
        {feats_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous}
    );
}

} // namespace mlx_lattice

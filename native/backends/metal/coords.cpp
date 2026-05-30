#include "backends/metal/coords.h"

#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx_lattice::metal {

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

mx::array make_offsets_array(const std::vector<Triple>& offsets) {
    std::vector<int32_t> flat;
    flat.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        flat.insert(flat.end(), offset.begin(), offset.end());
    }
    return mx::array(
        flat.begin(), mx::Shape{int(offsets.size()), 3}, mx::int32
    );
}

class SubmKernelMap : public mx::Primitive {
  public:
    SubmKernelMap(mx::Stream stream, int rows, int kernels)
        : mx::Primitive(stream), rows_(rows), kernels_(kernels) {}

    // MARK: - primitive

    void
    eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("SubmKernelMap has no CPU implementation.");
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
#ifdef _METAL_
        const auto& coords = inputs[0];
        const auto& offsets = inputs[1];
        auto& maps = outputs[0];
        auto& sizes = outputs[1];
        auto& kernels = outputs[2];

        maps.set_data(mx::allocator::malloc(maps.nbytes()));
        sizes.set_data(mx::allocator::malloc(sizes.nbytes()));
        kernels.set_data(mx::allocator::malloc(kernels.nbytes()));

        auto& s = stream();
        auto& device = mx::metal::device(s.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(s);

        int pair_slots = rows_ * kernels_;
        auto fill = device.get_kernel("fill_i32", library);

        encoder.set_compute_pipeline_state(fill);
        encoder.set_output_array(sizes, 0);
        int zero = 0;
        encoder.set_bytes(zero, 1);
        encoder.set_bytes(kernels_, 2);
        auto size_group = std::min(
            static_cast<size_t>(std::max(kernels_, 1)),
            fill->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(kernels_), 1, 1),
            MTL::Size(size_group, 1, 1)
        );

        encoder.set_compute_pipeline_state(fill);
        encoder.set_output_array(kernels, 0);
        int invalid = -1;
        encoder.set_bytes(invalid, 1);
        encoder.set_bytes(pair_slots, 2);
        auto kernel_group = std::min(
            static_cast<size_t>(std::max(pair_slots, 1)),
            fill->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(pair_slots), 1, 1),
            MTL::Size(kernel_group, 1, 1)
        );

        if (pair_slots == 0) {
            return;
        }

        auto build = device.get_kernel("build_subm_kernel_map_i32", library);
        encoder.set_compute_pipeline_state(build);
        encoder.set_input_array(coords, 0);
        encoder.set_input_array(offsets, 1);
        encoder.set_output_array(maps, 2);
        encoder.set_output_array(sizes, 3);
        encoder.set_output_array(kernels, 4);
        encoder.set_bytes(rows_, 5);
        encoder.set_bytes(kernels_, 6);
        auto build_group = std::min(
            static_cast<size_t>(pair_slots),
            build->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(pair_slots), 1, 1),
            MTL::Size(build_group, 1, 1)
        );
#else
        throw std::runtime_error("Metal support is not available.");
#endif
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("SubmKernelMap has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("SubmKernelMap has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("SubmKernelMap has no vmap implementation.");
    }

    const char* name() const override { return "SubmKernelMap"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& map = static_cast<const SubmKernelMap&>(other);
        return rows_ == map.rows_ && kernels_ == map.kernels_;
    }

  private:
    int rows_;
    int kernels_;
};

} // namespace

// MARK: - api

KernelMapData
build_subm_kernel_map(const mx::array& coords, Triple kernel_size) {
    if (coords.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal coordinate maps require int32 coords."
        );
    }

    auto offsets = kernel_offsets(kernel_size);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto pair_slots = rows * int(offsets.size());
    auto outputs = mx::array::make_arrays(
        {mx::Shape{pair_slots, 2},
         mx::Shape{int(offsets.size())},
         mx::Shape{pair_slots}},
        {mx::int32, mx::int32, mx::int32},
        std::make_shared<SubmKernelMap>(
            mx::default_stream(mx::Device::gpu), rows, int(offsets.size())
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(offset_values, false, mx::Device::gpu)}
    );

    return {
        outputs[0],
        outputs[1],
        outputs[2],
        coords,
        offset_values,
    };
}

} // namespace mlx_lattice::metal

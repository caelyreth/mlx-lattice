#include "backends/metal/coords.h"

#include <dlfcn.h>

#include <algorithm>
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

mx::array make_i32_array(std::vector<int32_t> data, mx::Shape shape) {
    return mx::array(data.begin(), std::move(shape), mx::int32);
}

int next_power_of_two(int value) {
    int out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

class SubmKernelMap : public mx::Primitive {
  public:
    SubmKernelMap(
        mx::Stream stream,
        int rows,
        int kernels,
        int center_kernel,
        int table_capacity
    )
        : mx::Primitive(stream), rows_(rows), kernels_(kernels),
          center_kernel_(center_kernel), table_capacity_(table_capacity) {}

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
        auto& residual_maps = outputs[3];
        auto& residual_kernels = outputs[4];
        auto& residual_offsets = outputs[5];
        auto& table_keys = outputs[6];
        auto& table_rows = outputs[7];

        maps.set_data(mx::allocator::malloc(maps.nbytes()));
        sizes.set_data(mx::allocator::malloc(sizes.nbytes()));
        kernels.set_data(mx::allocator::malloc(kernels.nbytes()));
        residual_maps.set_data(mx::allocator::malloc(residual_maps.nbytes()));
        residual_kernels.set_data(
            mx::allocator::malloc(residual_kernels.nbytes())
        );
        residual_offsets.set_data(
            mx::allocator::malloc(residual_offsets.nbytes())
        );
        table_keys.set_data(mx::allocator::malloc(table_keys.nbytes()));
        table_rows.set_data(mx::allocator::malloc(table_rows.nbytes()));

        auto& s = stream();
        auto& device = mx::metal::device(s.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(s);

        int pair_slots = rows_ * kernels_;
        auto fill = device.get_kernel("fill_i32", library);
        int invalid = -1;
        int empty_key = 0x7fffffff;

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

        int residual_slots = rows_ * std::max(kernels_ - 1, 0);
        encoder.set_compute_pipeline_state(fill);
        encoder.set_output_array(residual_kernels, 0);
        encoder.set_bytes(invalid, 1);
        encoder.set_bytes(residual_slots, 2);
        auto residual_group = std::min(
            static_cast<size_t>(std::max(residual_slots, 1)),
            fill->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(residual_slots), 1, 1),
            MTL::Size(residual_group, 1, 1)
        );

        encoder.set_compute_pipeline_state(fill);
        encoder.set_output_array(table_keys, 0);
        encoder.set_bytes(empty_key, 1);
        encoder.set_bytes(table_capacity_, 2);
        auto table_key_group = std::min(
            static_cast<size_t>(std::max(table_capacity_, 1)),
            fill->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(table_capacity_), 1, 1),
            MTL::Size(table_key_group, 1, 1)
        );

        encoder.set_compute_pipeline_state(fill);
        encoder.set_output_array(table_rows, 0);
        encoder.set_bytes(invalid, 1);
        encoder.set_bytes(table_capacity_, 2);
        auto table_row_group = std::min(
            static_cast<size_t>(std::max(table_capacity_, 1)),
            fill->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(table_capacity_), 1, 1),
            MTL::Size(table_row_group, 1, 1)
        );

        auto fill_linear = device.get_kernel("fill_linear_i32", library);
        encoder.set_compute_pipeline_state(fill_linear);
        encoder.set_output_array(residual_offsets, 0);
        int residual_step = std::max(kernels_ - 1, 0);
        int residual_offset_count = rows_ + 1;
        encoder.set_bytes(residual_step, 1);
        encoder.set_bytes(residual_offset_count, 2);
        auto offset_group = std::min(
            static_cast<size_t>(std::max(residual_offset_count, 1)),
            fill_linear->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(residual_offset_count), 1, 1),
            MTL::Size(offset_group, 1, 1)
        );

        if (pair_slots == 0) {
            return;
        }

        auto insert = device.get_kernel("insert_coord_hash_i32", library);
        encoder.set_compute_pipeline_state(insert);
        encoder.set_input_array(coords, 0);
        encoder.set_output_array(table_keys, 1);
        encoder.set_output_array(table_rows, 2);
        encoder.set_bytes(rows_, 3);
        encoder.set_bytes(table_capacity_, 4);
        encoder.set_bytes(empty_key, 5);
        auto insert_group = std::min(
            static_cast<size_t>(rows_), insert->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(rows_), 1, 1),
            MTL::Size(insert_group, 1, 1)
        );

        auto build = device.get_kernel("build_subm_kernel_map_i32", library);
        encoder.set_compute_pipeline_state(build);
        encoder.set_input_array(coords, 0);
        encoder.set_input_array(offsets, 1);
        encoder.set_input_array(table_keys, 2);
        encoder.set_input_array(table_rows, 3);
        encoder.set_output_array(maps, 4);
        encoder.set_output_array(sizes, 5);
        encoder.set_output_array(kernels, 6);
        encoder.set_output_array(residual_maps, 7);
        encoder.set_output_array(residual_kernels, 8);
        encoder.set_bytes(rows_, 9);
        encoder.set_bytes(kernels_, 10);
        encoder.set_bytes(center_kernel_, 11);
        encoder.set_bytes(table_capacity_, 12);
        encoder.set_bytes(empty_key, 13);
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
        return rows_ == map.rows_ && kernels_ == map.kernels_ &&
               center_kernel_ == map.center_kernel_ &&
               table_capacity_ == map.table_capacity_;
    }

  private:
    int rows_;
    int kernels_;
    int center_kernel_;
    int table_capacity_;
};

class GenerativeMap : public mx::Primitive {
  public:
    GenerativeMap(mx::Stream stream, int rows, int kernels, Triple stride)
        : mx::Primitive(stream), rows_(rows), kernels_(kernels),
          stride_(stride) {}

    // MARK: - primitive

    void
    eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("GenerativeMap has no CPU implementation.");
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
#ifdef _METAL_
        const auto& coords = inputs[0];
        const auto& offsets = inputs[1];
        auto& maps = outputs[0];
        auto& kernels = outputs[1];
        auto& out_coords = outputs[2];

        maps.set_data(mx::allocator::malloc(maps.nbytes()));
        kernels.set_data(mx::allocator::malloc(kernels.nbytes()));
        out_coords.set_data(mx::allocator::malloc(out_coords.nbytes()));

        auto& s = stream();
        auto& device = mx::metal::device(s.device);
        auto library = device.get_library("mlx_lattice", binary_dir());
        auto& encoder = mx::metal::get_command_encoder(s);
        auto build = device.get_kernel("build_generative_map_i32", library);
        int pair_count = rows_ * kernels_;
        if (pair_count == 0) {
            return;
        }

        encoder.set_compute_pipeline_state(build);
        encoder.set_input_array(coords, 0);
        encoder.set_input_array(offsets, 1);
        encoder.set_output_array(maps, 2);
        encoder.set_output_array(kernels, 3);
        encoder.set_output_array(out_coords, 4);
        encoder.set_bytes(rows_, 5);
        encoder.set_bytes(kernels_, 6);
        encoder.set_bytes(stride_[0], 7);
        encoder.set_bytes(stride_[1], 8);
        encoder.set_bytes(stride_[2], 9);
        auto group = std::min(
            static_cast<size_t>(pair_count),
            build->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(pair_count), 1, 1),
            MTL::Size(group, 1, 1)
        );
#else
        throw std::runtime_error("Metal support is not available.");
#endif
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("GenerativeMap has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("GenerativeMap has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("GenerativeMap has no vmap implementation.");
    }

    const char* name() const override { return "GenerativeMap"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& map = static_cast<const GenerativeMap&>(other);
        return rows_ == map.rows_ && kernels_ == map.kernels_ &&
               stride_ == map.stride_;
    }

  private:
    int rows_;
    int kernels_;
    Triple stride_;
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
    auto center = std::find(offsets.begin(), offsets.end(), Triple{0, 0, 0});
    if (center == offsets.end()) {
        throw std::invalid_argument(
            "submanifold maps require a kernel with center offset."
        );
    }
    int center_kernel = int(std::distance(offsets.begin(), center));
    auto rows = coords.shape(0);
    auto pair_slots = rows * int(offsets.size());
    auto residual_slots = rows * std::max(int(offsets.size()) - 1, 0);
    auto table_capacity = next_power_of_two(std::max(rows * 4, 1));
    auto outputs = mx::array::make_arrays(
        {mx::Shape{pair_slots, 2},
         mx::Shape{int(offsets.size())},
         mx::Shape{pair_slots},
         mx::Shape{residual_slots, 2},
         mx::Shape{residual_slots},
         mx::Shape{rows + 1},
         mx::Shape{table_capacity},
         mx::Shape{table_capacity}},
        {mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32},
        std::make_shared<SubmKernelMap>(
            mx::default_stream(mx::Device::gpu),
            rows,
            int(offsets.size()),
            center_kernel,
            table_capacity
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(offset_values, false, mx::Device::gpu)}
    );

    return {
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        coords,
        offset_values,
    };
}

KernelMapData build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
) {
    if (coords.dtype() != mx::int32) {
        throw std::invalid_argument(
            "Metal generative maps require int32 coords."
        );
    }

    auto offsets = kernel_offsets(kernel_size);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto pair_count = rows * int(offsets.size());
    auto sizes =
        mx::full({int(offsets.size())}, rows, mx::int32, mx::Device::gpu);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{pair_count, 2},
         mx::Shape{pair_count},
         mx::Shape{pair_count, 4}},
        {mx::int32, mx::int32, mx::int32},
        std::make_shared<GenerativeMap>(
            mx::default_stream(mx::Device::gpu),
            rows,
            int(offsets.size()),
            stride
        ),
        {mx::contiguous(coords, false, mx::Device::gpu),
         mx::contiguous(offset_values, false, mx::Device::gpu)}
    );

    return {
        outputs[0],
        sizes,
        outputs[1],
        make_i32_array({}, mx::Shape{0, 2}),
        make_i32_array({}, mx::Shape{0}),
        make_i32_array({0}, mx::Shape{1}),
        outputs[2],
        offset_values,
    };
}

} // namespace mlx_lattice::metal

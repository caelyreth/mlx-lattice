#include "backends/metal/pool/runtime.h"

#include <algorithm>
#include <stdexcept>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "mlx/device.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::pool {
namespace {

int reduce_id(PoolReduceOp op) {
    switch (op) {
    case PoolReduceOp::Sum:
        return 0;
    case PoolReduceOp::Max:
        return 1;
    case PoolReduceOp::Avg:
        return 2;
    }
}

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}

bool is_identity(Triple stride, Triple padding) {
    return stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
}

bool is_nonoverlapping(SparsePoolShape shape, Triple stride, Triple padding) {
    return padding == Triple{0, 0, 0} && stride[0] > 0 && stride[1] > 0 &&
           stride[2] > 0 &&
           shape.n_kernels == stride[0] * stride[1] * stride[2];
}

struct PoolGeometry {
    Triple stride;
    Triple padding;
};

#ifdef _METAL_
mx::array make_int32_temp(int elements) {
    auto count = std::max(elements, 1);
    return mx::array(
        mx::allocator::malloc(static_cast<size_t>(count) * sizeof(int32_t)),
        mx::Shape{count},
        mx::int32
    );
}

int next_power_of_two(int value) {
    auto out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, size_t elements) {
    auto threads = std::max<size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}

template <typename Encoder>
void bind_autodiff_shape(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const PoolGeometry& geometry
) {
    encoder.set_bytes(reduce_id(reduce), 7);
    encoder.set_bytes(shape.in_capacity, 8);
    encoder.set_bytes(shape.out_capacity, 9);
    encoder.set_bytes(shape.n_kernels, 10);
    encoder.set_bytes(shape.channels, 11);
    encoder.set_bytes(geometry.stride[0], 12);
    encoder.set_bytes(geometry.stride[1], 13);
    encoder.set_bytes(geometry.stride[2], 14);
    encoder.set_bytes(geometry.padding[0], 15);
    encoder.set_bytes(geometry.padding[1], 16);
    encoder.set_bytes(geometry.padding[2], 17);
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[1], 0), 20);
    encoder.set_bytes(stride_at(inputs[1], 1), 21);
    encoder.set_bytes(stride_at(inputs[2], 0), 22);
    encoder.set_bytes(stride_at(inputs[2], 1), 23);
}
#endif

} // namespace

bool is_supported(
    const mx::array& coords, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets
) {
#if MLX_LATTICE_HAS_METAL
    return mx::is_available(mx::Device::gpu) && coords.dtype() == mx::int32 &&
           active_rows.dtype() == mx::int32 && feats.dtype() == mx::float32 &&
           offsets.dtype() == mx::int32;
#else
    (void)coords;
    (void)active_rows;
    (void)feats;
    (void)offsets;
    return false;
#endif
}

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    allocate_all(outputs);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto coord_elements = static_cast<size_t>(shape.out_capacity) * 4;
    auto feat_elements = static_cast<size_t>(shape.out_capacity) *
                         static_cast<size_t>(shape.channels);

    auto clear = device.get_kernel("sparse_pool_clear_f32_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[SparseOutCoords], 0);
    encoder.set_output_array(outputs[SparseOutFeats], 1);
    encoder.set_output_array(outputs[SparseCounts], 2);
    encoder.set_bytes(static_cast<int>(coord_elements), 3);
    encoder.set_bytes(static_cast<int>(feat_elements), 4);
    encoder.set_bytes(reduce_id(reduce), 5);
    dispatch_1d(encoder, clear, std::max(coord_elements, feat_elements));

    auto downsample = is_nonoverlapping(shape, stride, padding);
    if (downsample) {
        auto table_capacity =
            next_power_of_two(std::max(shape.in_capacity * 2, 1));
        auto empty_key = int(0x7fffffff);
        auto table_keys = make_int32_temp(table_capacity);
        auto table_rows = make_int32_temp(table_capacity);
        auto kernel = device.get_kernel(
            "sparse_pool_downsample_coords_hash_i32", library
        );
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(table_keys, 2);
        encoder.set_output_array(table_rows, 3);
        encoder.set_output_array(outputs[SparseOutCoords], 4);
        encoder.set_output_array(outputs[SparseCounts], 5);
        encoder.set_bytes(shape.in_capacity, 6);
        encoder.set_bytes(table_capacity, 7);
        encoder.set_bytes(empty_key, 8);
        encoder.set_bytes(stride[0], 9);
        encoder.set_bytes(stride[1], 10);
        encoder.set_bytes(stride[2], 11);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    } else {
        auto identity = is_identity(stride, padding);
        auto kernel = device.get_kernel(
            identity ? "sparse_pool_identity_coords_i32"
                     : "sparse_pool_forward_coords_i32",
            library
        );
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[3], 2);
        encoder.set_output_array(outputs[SparseOutCoords], 3);
        encoder.set_output_array(outputs[SparseCounts], 4);
        encoder.set_bytes(shape.in_capacity, 5);
        encoder.set_bytes(shape.n_kernels, 6);
        if (!identity) {
            encoder.set_bytes(stride[0], 7);
            encoder.set_bytes(stride[1], 8);
            encoder.set_bytes(stride[2], 9);
            encoder.set_bytes(padding[0], 10);
            encoder.set_bytes(padding[1], 11);
            encoder.set_bytes(padding[2], 12);
        }
        dispatch_1d(
            encoder,
            kernel,
            identity ? static_cast<size_t>(shape.in_capacity) * 4
                     : static_cast<size_t>(shape.in_capacity)
        );
    }

    auto row_gather = downsample && reduce != PoolReduceOp::Sum;
    auto gather = device.get_kernel(
        downsample ? (row_gather ? "sparse_pool_downsample_gather_rows_f32_i32"
                                 : "sparse_pool_downsample_gather_f32_i32")
                   : "sparse_pool_forward_gather_f32_i32",
        library
    );
    encoder.set_compute_pipeline_state(gather);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    if (downsample) {
        encoder.set_input_array(outputs[SparseOutCoords], 3);
        encoder.set_input_array(outputs[SparseCounts], 4);
        encoder.set_output_array(outputs[SparseOutFeats], 5);
        encoder.set_bytes(reduce_id(reduce), 6);
        encoder.set_bytes(shape.in_capacity, 7);
        encoder.set_bytes(shape.out_capacity, 8);
        encoder.set_bytes(shape.channels, 9);
        encoder.set_bytes(stride[0], 10);
        encoder.set_bytes(stride[1], 11);
        encoder.set_bytes(stride[2], 12);
        encoder.set_bytes(stride_at(inputs[2], 0), 13);
        encoder.set_bytes(stride_at(inputs[2], 1), 14);
    } else {
        encoder.set_input_array(inputs[3], 3);
        encoder.set_input_array(outputs[SparseOutCoords], 4);
        encoder.set_input_array(outputs[SparseCounts], 5);
        encoder.set_output_array(outputs[SparseOutFeats], 6);
        encoder.set_bytes(reduce_id(reduce), 7);
        encoder.set_bytes(shape.in_capacity, 8);
        encoder.set_bytes(shape.out_capacity, 9);
        encoder.set_bytes(shape.n_kernels, 10);
        encoder.set_bytes(shape.channels, 11);
        encoder.set_bytes(stride[0], 12);
        encoder.set_bytes(stride[1], 13);
        encoder.set_bytes(stride[2], 14);
        encoder.set_bytes(padding[0], 15);
        encoder.set_bytes(padding[1], 16);
        encoder.set_bytes(padding[2], 17);
        encoder.set_bytes(stride_at(inputs[2], 0), 18);
        encoder.set_bytes(stride_at(inputs[2], 1), 19);
    }
    dispatch_1d(
        encoder,
        gather,
        row_gather ? static_cast<size_t>(shape.out_capacity) : feat_elements
    );
#else
    (void)reduce;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_pool_grad_f32_i32_serial", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 6);
    bind_autodiff_shape(
        encoder, inputs, reduce, shape, PoolGeometry{stride, padding}
    );
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)reduce;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_pool_jvp_f32_i32_serial", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 6);
    bind_autodiff_shape(
        encoder, inputs, reduce, shape, PoolGeometry{stride, padding}
    );
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)reduce;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::pool

#include "backends/metal/exec/runtime.h"

#include <stdexcept>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "mlx/device.h"
#include "mlx/stream.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::exec::metal {

namespace {

int map_op_id(SparseMapOp op) {
    switch (op) {
    case SparseMapOp::Forward:
        return 0;
    case SparseMapOp::Transposed:
        return 1;
    case SparseMapOp::Generative:
        return 2;
    }
}

int pool_op_id(PoolReduceOp op) {
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

bool metal_runtime_available() {
#if MLX_LATTICE_HAS_METAL
    return mx::is_available(mx::Device::gpu);
#else
    return false;
#endif
}

} // namespace

bool can_run_sparse_conv(
    const mx::array& coords, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets
) {
#if MLX_LATTICE_HAS_METAL
    return metal_runtime_available() && coords.dtype() == mx::int32 &&
           active_rows.dtype() == mx::int32 && feats.dtype() == mx::float32 &&
           weights.dtype() == mx::float32 && offsets.dtype() == mx::int32;
#else
    (void)coords;
    (void)active_rows;
    (void)feats;
    (void)weights;
    (void)offsets;
    return false;
#endif
}

bool can_run_sparse_pool(
    const mx::array& coords, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets
) {
#if MLX_LATTICE_HAS_METAL
    return metal_runtime_available() && coords.dtype() == mx::int32 &&
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

void eval_sparse_conv(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_conv_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(outputs[SparseOutCoords], 5);
    encoder.set_output_array(outputs[SparseOutFeats], 6);
    encoder.set_output_array(outputs[SparseCounts], 7);
    encoder.set_bytes(map_op_id(op), 8);
    encoder.set_bytes(shape.in_capacity, 9);
    encoder.set_bytes(shape.out_capacity, 10);
    encoder.set_bytes(shape.n_kernels, 11);
    encoder.set_bytes(shape.in_channels, 12);
    encoder.set_bytes(shape.out_channels, 13);
    encoder.set_bytes(stride[0], 14);
    encoder.set_bytes(stride[1], 15);
    encoder.set_bytes(stride[2], 16);
    encoder.set_bytes(padding[0], 17);
    encoder.set_bytes(padding[1], 18);
    encoder.set_bytes(padding[2], 19);
    encoder.set_bytes(stride_at(inputs[2], 0), 20);
    encoder.set_bytes(stride_at(inputs[2], 1), 21);
    encoder.set_bytes(stride_at(inputs[3], 0), 22);
    encoder.set_bytes(stride_at(inputs[3], 1), 23);
    encoder.set_bytes(stride_at(inputs[3], 2), 24);
    encoder.set_bytes(inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 25);
    encoder.set_bytes(inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 26);
    encoder.set_bytes(shape.weight_layout, 27);
    encoder.set_bytes(shape.kernel_x, 28);
    encoder.set_bytes(shape.kernel_y, 29);
    encoder.set_bytes(shape.kernel_z, 30);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_conv_input_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel =
        device.get_kernel("sparse_conv_input_grad_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 5);
    encoder.set_bytes(map_op_id(op), 6);
    encoder.set_bytes(shape.in_capacity, 7);
    encoder.set_bytes(shape.out_capacity, 8);
    encoder.set_bytes(shape.n_kernels, 9);
    encoder.set_bytes(shape.in_channels, 10);
    encoder.set_bytes(shape.out_channels, 11);
    encoder.set_bytes(stride[0], 12);
    encoder.set_bytes(stride[1], 13);
    encoder.set_bytes(stride[2], 14);
    encoder.set_bytes(padding[0], 15);
    encoder.set_bytes(padding[1], 16);
    encoder.set_bytes(padding[2], 17);
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[3], 0), 20);
    encoder.set_bytes(stride_at(inputs[3], 1), 21);
    encoder.set_bytes(stride_at(inputs[3], 2), 22);
    encoder.set_bytes(inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 23);
    encoder.set_bytes(inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 24);
    encoder.set_bytes(shape.weight_layout, 25);
    encoder.set_bytes(shape.kernel_x, 26);
    encoder.set_bytes(shape.kernel_y, 27);
    encoder.set_bytes(shape.kernel_z, 28);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_conv_weight_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel =
        device.get_kernel("sparse_conv_weight_grad_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 5);
    encoder.set_bytes(map_op_id(op), 6);
    encoder.set_bytes(shape.in_capacity, 7);
    encoder.set_bytes(shape.out_capacity, 8);
    encoder.set_bytes(shape.n_kernels, 9);
    encoder.set_bytes(shape.in_channels, 10);
    encoder.set_bytes(shape.out_channels, 11);
    encoder.set_bytes(stride[0], 12);
    encoder.set_bytes(stride[1], 13);
    encoder.set_bytes(stride[2], 14);
    encoder.set_bytes(padding[0], 15);
    encoder.set_bytes(padding[1], 16);
    encoder.set_bytes(padding[2], 17);
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[1], 0), 20);
    encoder.set_bytes(stride_at(inputs[1], 1), 21);
    encoder.set_bytes(shape.weight_layout, 22);
    encoder.set_bytes(shape.kernel_x, 23);
    encoder.set_bytes(shape.kernel_y, 24);
    encoder.set_bytes(shape.kernel_z, 25);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_pool(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_pool_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(outputs[SparseOutCoords], 4);
    encoder.set_output_array(outputs[SparseOutFeats], 5);
    encoder.set_output_array(outputs[SparseCounts], 6);
    encoder.set_bytes(pool_op_id(reduce), 7);
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

void eval_sparse_pool_grad(
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
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_pool_grad_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 6);
    encoder.set_bytes(pool_op_id(reduce), 7);
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
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[1], 0), 20);
    encoder.set_bytes(stride_at(inputs[1], 1), 21);
    encoder.set_bytes(stride_at(inputs[2], 0), 22);
    encoder.set_bytes(stride_at(inputs[2], 1), 23);
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

void eval_sparse_pool_jvp(
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
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_pool_jvp_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 6);
    encoder.set_bytes(pool_op_id(reduce), 7);
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
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[1], 0), 20);
    encoder.set_bytes(stride_at(inputs[1], 1), 21);
    encoder.set_bytes(stride_at(inputs[2], 0), 22);
    encoder.set_bytes(stride_at(inputs[2], 1), 23);
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

} // namespace mlx_lattice::exec::metal

#include "backends/cpu/exec/algorithms.h"

#include "backends/cpu/exec/planning.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "backends/array_utils.h"
#include "backends/cpu/schedule.h"

namespace mlx_lattice::exec::cpu {

namespace {

void fill_zero(mx::array& out) {
    auto data = out.data<float>();
    std::fill(data, data + out.size(), 0.0F);
}

void fill_pool_init(mx::array& out, PoolReduceOp reduce) {
    auto data = out.data<float>();
    auto value = reduce == PoolReduceOp::Max
                     ? -std::numeric_limits<float>::infinity()
                     : 0.0F;
    std::fill(data, data + out.size(), value);
}

std::ptrdiff_t weight_offset(
    const mx::array& weights,
    const SparseConvShape& shape,
    int kernel,
    int in_channel,
    int out_channel
) {
    if (shape.weight_layout == 0) {
        return static_cast<std::ptrdiff_t>(kernel) * weights.strides(0) +
               static_cast<std::ptrdiff_t>(in_channel) * weights.strides(1) +
               static_cast<std::ptrdiff_t>(out_channel) * weights.strides(2);
    }

    auto xy = shape.kernel_y * shape.kernel_z;
    auto kx = kernel / xy;
    auto rem = kernel % xy;
    auto ky = rem / shape.kernel_z;
    auto kz = rem % shape.kernel_z;
    return static_cast<std::ptrdiff_t>(out_channel) * weights.strides(0) +
           static_cast<std::ptrdiff_t>(kx) * weights.strides(1) +
           static_cast<std::ptrdiff_t>(ky) * weights.strides(2) +
           static_cast<std::ptrdiff_t>(kz) * weights.strides(3) +
           static_cast<std::ptrdiff_t>(in_channel) * weights.strides(4);
}

std::ptrdiff_t dense_weight_offset(
    const SparseConvShape& shape,
    int kernel,
    int in_channel,
    int out_channel
) {
    if (shape.weight_layout == 0) {
        return (static_cast<std::ptrdiff_t>(kernel) * shape.in_channels +
                in_channel) *
                   shape.out_channels +
               out_channel;
    }

    auto xy = shape.kernel_y * shape.kernel_z;
    auto kx = kernel / xy;
    auto rem = kernel % xy;
    auto ky = rem / shape.kernel_z;
    auto kz = rem % shape.kernel_z;
    return (((static_cast<std::ptrdiff_t>(out_channel) * shape.kernel_x + kx) *
                 shape.kernel_y +
             ky) *
                shape.kernel_z +
            kz) *
               shape.in_channels +
           in_channel;
}

} // namespace

void eval_sparse_conv(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& coords = ready[0];
            const auto& active_rows = ready[1];
            const auto& feats = ready[2];
            const auto& weights = ready[3];
            const auto& offsets = ready[4];

            auto plan =
                build_plan(op, coords, active_rows, offsets, stride, padding);
            write_coords(task_outputs[SparseOutCoords], plan.out_coords);
            write_counts(task_outputs[SparseCounts], plan);

            auto& out = task_outputs[SparseOutFeats];
            fill_zero(out);
            auto* out_data = out.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* weight_data = weights.data<float>();
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);

            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto kernel = edge[2];
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.out_channels;
                for (int ci = 0; ci < shape.in_channels; ++ci) {
                    const auto value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(ci) * feat_s1];
                    for (int co = 0; co < shape.out_channels; ++co) {
                        out_row_data[co] +=
                            value * weight_data[weight_offset(
                                        weights, shape, kernel, ci, co
                                    )];
                    }
                }
            }
        }
    );
}

void eval_sparse_conv_input_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& cotangent = ready[0];
            const auto& coords = ready[1];
            const auto& active_rows = ready[2];
            const auto& weights = ready[3];
            const auto& offsets = ready[4];

            auto plan =
                build_plan(op, coords, active_rows, offsets, stride, padding);
            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto* weight_data = weights.data<float>();
            const auto cotangent_s0 = cotangent.strides(0);
            const auto cotangent_s1 = cotangent.strides(1);

            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto kernel = edge[2];
                auto* grad_row =
                    grad_data +
                    static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
                for (int ci = 0; ci < shape.in_channels; ++ci) {
                    for (int co = 0; co < shape.out_channels; ++co) {
                        auto cotangent_index =
                            static_cast<std::ptrdiff_t>(out_row) *
                                cotangent_s0 +
                            static_cast<std::ptrdiff_t>(co) * cotangent_s1;
                        grad_row[ci] += cotangent_data[cotangent_index] *
                                        weight_data[weight_offset(
                                            weights, shape, kernel, ci, co
                                        )];
                    }
                }
            }
        }
    );
}

void eval_sparse_conv_weight_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& feats = ready[0];
            const auto& cotangent = ready[1];
            const auto& coords = ready[2];
            const auto& active_rows = ready[3];
            const auto& offsets = ready[4];

            auto plan =
                build_plan(op, coords, active_rows, offsets, stride, padding);
            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            const auto cotangent_s0 = cotangent.strides(0);
            const auto cotangent_s1 = cotangent.strides(1);

            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto kernel = edge[2];
                for (int ci = 0; ci < shape.in_channels; ++ci) {
                    for (int co = 0; co < shape.out_channels; ++co) {
                        auto feat_index =
                            static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                            static_cast<std::ptrdiff_t>(ci) * feat_s1;
                        auto cotangent_index =
                            static_cast<std::ptrdiff_t>(out_row) *
                                cotangent_s0 +
                            static_cast<std::ptrdiff_t>(co) * cotangent_s1;
                        grad_data[dense_weight_offset(shape, kernel, ci, co)] +=
                            feat_data[feat_index] *
                            cotangent_data[cotangent_index];
                    }
                }
            }
        }
    );
}

void eval_sparse_pool(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& coords = ready[0];
            const auto& active_rows = ready[1];
            const auto& feats = ready[2];
            const auto& offsets = ready[3];

            auto plan = build_plan(
                SparseMapOp::Forward,
                coords,
                active_rows,
                offsets,
                stride,
                padding
            );
            write_coords(task_outputs[SparseOutCoords], plan.out_coords);
            write_counts(task_outputs[SparseCounts], plan);

            auto& out = task_outputs[SparseOutFeats];
            fill_pool_init(out, reduce);
            auto* out_data = out.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.channels;
                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(channel) * feat_s1];
                    if (reduce == PoolReduceOp::Max) {
                        out_row_data[channel] =
                            std::max(out_row_data[channel], value);
                    } else {
                        out_row_data[channel] += value;
                    }
                }
            }

            if (reduce == PoolReduceOp::Avg) {
                auto degrees = pool_degrees(plan, shape.out_capacity);
                for (int row = 0; row < int(plan.out_coords.size()); ++row) {
                    auto denom = std::max(degrees[row], int32_t{1});
                    auto* out_row =
                        out_data +
                        static_cast<std::ptrdiff_t>(row) * shape.channels;
                    for (int channel = 0; channel < shape.channels; ++channel) {
                        out_row[channel] /= float(denom);
                    }
                }
            }
        }
    );
}

void eval_sparse_pool_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& cotangent = ready[0];
            const auto& feats = ready[1];
            const auto& pooled = ready[2];
            const auto& coords = ready[3];
            const auto& active_rows = ready[4];
            const auto& offsets = ready[5];

            auto plan = build_plan(
                SparseMapOp::Forward,
                coords,
                active_rows,
                offsets,
                stride,
                padding
            );
            auto degrees = pool_degrees(plan, shape.out_capacity);
            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* pooled_data = pooled.data<float>();
            const auto cotangent_s0 = cotangent.strides(0);
            const auto cotangent_s1 = cotangent.strides(1);
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            const auto pooled_s0 = pooled.strides(0);
            const auto pooled_s1 = pooled.strides(1);

            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto* grad_row =
                    grad_data +
                    static_cast<std::ptrdiff_t>(in_row) * shape.channels;
                auto denom = std::max(degrees[out_row], int32_t{1});

                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto feat_value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(channel) * feat_s1];
                    auto pooled_value = pooled_data
                        [static_cast<std::ptrdiff_t>(out_row) * pooled_s0 +
                         static_cast<std::ptrdiff_t>(channel) * pooled_s1];
                    if (reduce == PoolReduceOp::Max &&
                        feat_value != pooled_value) {
                        continue;
                    }
                    auto scale = reduce == PoolReduceOp::Avg
                                     ? 1.0F / float(denom)
                                     : 1.0F;
                    auto cotangent_value = cotangent_data
                        [static_cast<std::ptrdiff_t>(out_row) * cotangent_s0 +
                         static_cast<std::ptrdiff_t>(channel) * cotangent_s1];
                    grad_row[channel] += cotangent_value * scale;
                }
            }
        }
    );
}

void eval_sparse_pool_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& tangent = ready[0];
            const auto& feats = ready[1];
            const auto& pooled = ready[2];
            const auto& coords = ready[3];
            const auto& active_rows = ready[4];
            const auto& offsets = ready[5];

            auto plan = build_plan(
                SparseMapOp::Forward,
                coords,
                active_rows,
                offsets,
                stride,
                padding
            );
            auto degrees = pool_degrees(plan, shape.out_capacity);
            auto& out = task_outputs[0];
            fill_zero(out);
            auto* out_data = out.data<float>();
            const auto* tangent_data = tangent.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* pooled_data = pooled.data<float>();
            const auto tangent_s0 = tangent.strides(0);
            const auto tangent_s1 = tangent.strides(1);
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            const auto pooled_s0 = pooled.strides(0);
            const auto pooled_s1 = pooled.strides(1);

            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.channels;
                auto denom = std::max(degrees[out_row], int32_t{1});

                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto feat_value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(channel) * feat_s1];
                    auto pooled_value = pooled_data
                        [static_cast<std::ptrdiff_t>(out_row) * pooled_s0 +
                         static_cast<std::ptrdiff_t>(channel) * pooled_s1];
                    if (reduce == PoolReduceOp::Max &&
                        feat_value != pooled_value) {
                        continue;
                    }
                    auto scale = reduce == PoolReduceOp::Avg
                                     ? 1.0F / float(denom)
                                     : 1.0F;
                    auto tangent_value = tangent_data
                        [static_cast<std::ptrdiff_t>(in_row) * tangent_s0 +
                         static_cast<std::ptrdiff_t>(channel) * tangent_s1];
                    out_row_data[channel] += tangent_value * scale;
                }
            }
        }
    );
}

} // namespace mlx_lattice::exec::cpu

#include "backends/cpu/pool/algorithms.h"

#include "backends/cpu/pool/planning.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backends/array_utils.h"
#include "backends/cpu/schedule.h"

namespace mlx_lattice::backend::cpu::pool {
namespace {

void fill_zero(mx::array& out) {
    auto* data = out.data<float>();
    std::fill(data, data + out.size(), 0.0F);
}

void fill_reduction_identity(mx::array& out, PoolReduceOp reduce) {
    auto* data = out.data<float>();
    auto value = reduce == PoolReduceOp::Max
                     ? -std::numeric_limits<float>::infinity()
                     : 0.0F;
    std::fill(data, data + out.size(), value);
}

int edge_rank(const Edge& edge, const SparsePoolShape& shape) {
    return edge[0] * shape.n_kernels + edge[2];
}

std::size_t channel_key(int out_row, int channel, int channels) {
    return static_cast<std::size_t>(out_row) *
               static_cast<std::size_t>(channels) +
           static_cast<std::size_t>(channel);
}

struct MaxTiePolicy {
    std::vector<int32_t> counts;
    std::vector<int32_t> first_ranks;
};

MaxTiePolicy build_max_tie_policy(
    const Plan& plan,
    const mx::array& feats,
    const mx::array& pooled,
    SparsePoolShape shape
) {
    auto size = static_cast<std::size_t>(shape.out_capacity) *
                static_cast<std::size_t>(shape.channels);
    MaxTiePolicy policy{
        std::vector<int32_t>(size, 0),
        std::vector<int32_t>(size, std::numeric_limits<int32_t>::max()),
    };
    const auto* feat_data = feats.data<float>();
    const auto* pooled_data = pooled.data<float>();

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto rank = edge_rank(edge, shape);
        for (int channel = 0; channel < shape.channels; ++channel) {
            auto feat_value = feat_data
                [static_cast<std::ptrdiff_t>(in_row) * feats.strides(0) +
                 static_cast<std::ptrdiff_t>(channel) * feats.strides(1)];
            auto pooled_value = pooled_data
                [static_cast<std::ptrdiff_t>(out_row) * pooled.strides(0) +
                 static_cast<std::ptrdiff_t>(channel) * pooled.strides(1)];
            if (feat_value != pooled_value) {
                continue;
            }
            auto key = channel_key(out_row, channel, shape.channels);
            ++policy.counts[key];
            policy.first_ranks[key] = std::min(policy.first_ranks[key], rank);
        }
    }
    return policy;
}

} // namespace

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            auto plan =
                build_plan(ready[0], ready[1], ready[3], stride, padding);
            write_coords(task_outputs[SparseOutCoords], plan.out_coords);
            write_counts(task_outputs[SparseCounts], plan);

            const auto& feats = ready[2];
            auto& out = task_outputs[SparseOutFeats];
            fill_reduction_identity(out, reduce);
            auto* out_data = out.data<float>();
            const auto* feat_data = feats.data<float>();
            for (auto edge : plan.edges) {
                auto* out_row =
                    out_data +
                    static_cast<std::ptrdiff_t>(edge[1]) * shape.channels;
                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto value = feat_data
                        [static_cast<std::ptrdiff_t>(edge[0]) *
                             feats.strides(0) +
                         static_cast<std::ptrdiff_t>(channel) *
                             feats.strides(1)];
                    if (reduce == PoolReduceOp::Max) {
                        out_row[channel] = std::max(out_row[channel], value);
                    } else {
                        out_row[channel] += value;
                    }
                }
            }

            if (reduce != PoolReduceOp::Avg) {
                return;
            }
            auto row_degrees = degrees(plan, shape.out_capacity);
            for (int row = 0; row < int(plan.out_coords.size()); ++row) {
                auto scale =
                    1.0F / float(std::max(row_degrees[row], int32_t{1}));
                auto* out_row = out_data + static_cast<std::ptrdiff_t>(row) *
                                               shape.channels;
                for (int channel = 0; channel < shape.channels; ++channel) {
                    out_row[channel] *= scale;
                }
            }
        }
    );
}

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
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
            auto plan =
                build_plan(ready[3], ready[4], ready[5], stride, padding);
            auto row_degrees = degrees(plan, shape.out_capacity);
            auto ties = reduce == PoolReduceOp::Max
                            ? build_max_tie_policy(plan, feats, pooled, shape)
                            : MaxTiePolicy{};

            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* pooled_data = pooled.data<float>();
            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto* grad_row =
                    grad_data +
                    static_cast<std::ptrdiff_t>(in_row) * shape.channels;
                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto scale = 1.0F;
                    if (reduce == PoolReduceOp::Max) {
                        auto feat_value = feat_data
                            [static_cast<std::ptrdiff_t>(in_row) *
                                 feats.strides(0) +
                             static_cast<std::ptrdiff_t>(channel) *
                                 feats.strides(1)];
                        auto pooled_value = pooled_data
                            [static_cast<std::ptrdiff_t>(out_row) *
                                 pooled.strides(0) +
                             static_cast<std::ptrdiff_t>(channel) *
                                 pooled.strides(1)];
                        if (feat_value != pooled_value) {
                            continue;
                        }
                        auto count = ties.counts[channel_key(
                            out_row, channel, shape.channels
                        )];
                        if (count == 0) {
                            continue;
                        }
                        scale = 1.0F / float(count);
                    } else if (reduce == PoolReduceOp::Avg) {
                        scale =
                            1.0F /
                            float(std::max(row_degrees[out_row], int32_t{1}));
                    }
                    grad_row[channel] +=
                        cotangent_data
                            [static_cast<std::ptrdiff_t>(out_row) *
                                 cotangent.strides(0) +
                             static_cast<std::ptrdiff_t>(channel) *
                                 cotangent.strides(1)] *
                        scale;
                }
            }
        }
    );
}

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
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
            auto plan =
                build_plan(ready[3], ready[4], ready[5], stride, padding);
            auto row_degrees = degrees(plan, shape.out_capacity);
            auto ties = reduce == PoolReduceOp::Max
                            ? build_max_tie_policy(plan, feats, pooled, shape)
                            : MaxTiePolicy{};

            auto& out = task_outputs[0];
            fill_zero(out);
            auto* out_data = out.data<float>();
            const auto* tangent_data = tangent.data<float>();
            for (auto edge : plan.edges) {
                auto out_row = edge[1];
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.channels;
                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto scale = 1.0F;
                    if (reduce == PoolReduceOp::Max) {
                        auto key =
                            channel_key(out_row, channel, shape.channels);
                        if (edge_rank(edge, shape) != ties.first_ranks[key]) {
                            continue;
                        }
                    } else if (reduce == PoolReduceOp::Avg) {
                        scale =
                            1.0F /
                            float(std::max(row_degrees[out_row], int32_t{1}));
                    }
                    out_row_data[channel] +=
                        tangent_data
                            [static_cast<std::ptrdiff_t>(edge[0]) *
                                 tangent.strides(0) +
                             static_cast<std::ptrdiff_t>(channel) *
                                 tangent.strides(1)] *
                        scale;
                }
            }
        }
    );
}

} // namespace mlx_lattice::backend::cpu::pool

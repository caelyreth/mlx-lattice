#include "backends/cpu/exec/algorithms.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mlx_lattice::exec::cpu {

void eval_spmm_edges(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& feats = inputs[0];
    const auto& weights = inputs[1];
    const auto& in_rows = inputs[2];
    const auto& out_rows = inputs[3];
    const auto& kernel_ids = inputs[4];

    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);

    const auto* feat_data = feats.data<float>();
    const auto* weight_data = weights.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();

    const auto in_channels = shape.in_channels;
    const auto out_channels = shape.out_channels;
    for (int edge = 0; edge < shape.edge_count; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        const auto kernel = kernel_data[edge];
        if (in_row < 0 || in_row >= feats.shape(0) || out_row < 0 ||
            out_row >= shape.n_out_rows || kernel < 0 ||
            kernel >= weights.shape(0)) {
            throw std::out_of_range("spmm_edges row index is out of bounds.");
        }

        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * in_channels;
        const auto* weight_kernel =
            weight_data +
            static_cast<std::ptrdiff_t>(kernel) * in_channels * out_channels;
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * out_channels;

        for (int ci = 0; ci < in_channels; ++ci) {
            const auto value = feat_row[ci];
            const auto* weight_row =
                weight_kernel + static_cast<std::ptrdiff_t>(ci) * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                out_row_data[co] += value * weight_row[co];
            }
        }
    }
}

void eval_spmm_edges_input_grad(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& cotangent = inputs[0];
    const auto& weights = inputs[1];
    const auto& in_rows = inputs[2];
    const auto& out_rows = inputs[3];
    const auto& kernel_ids = inputs[4];

    auto& grad = outputs[0];
    grad.set_data(mx::allocator::malloc(grad.nbytes()));
    auto grad_data = grad.data<float>();
    std::fill(grad_data, grad_data + grad.size(), 0.0F);

    const auto* cotangent_data = cotangent.data<float>();
    const auto* weight_data = weights.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();

    for (int edge = 0; edge < shape.edge_count; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        const auto kernel = kernel_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows || kernel < 0 ||
            kernel >= shape.n_kernels) {
            throw std::out_of_range(
                "spmm_edges input gradient row index is out of bounds."
            );
        }

        auto* grad_row =
            grad_data + static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
        const auto* cotangent_row =
            cotangent_data +
            static_cast<std::ptrdiff_t>(out_row) * shape.out_channels;
        const auto* weight_kernel =
            weight_data + static_cast<std::ptrdiff_t>(kernel) *
                              shape.in_channels * shape.out_channels;

        for (int ci = 0; ci < shape.in_channels; ++ci) {
            const auto* weight_row =
                weight_kernel +
                static_cast<std::ptrdiff_t>(ci) * shape.out_channels;
            for (int co = 0; co < shape.out_channels; ++co) {
                grad_row[ci] += cotangent_row[co] * weight_row[co];
            }
        }
    }
}

void eval_spmm_edges_weight_grad(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& feats = inputs[0];
    const auto& cotangent = inputs[1];
    const auto& in_rows = inputs[2];
    const auto& out_rows = inputs[3];
    const auto& kernel_ids = inputs[4];

    auto& grad = outputs[0];
    grad.set_data(mx::allocator::malloc(grad.nbytes()));
    auto grad_data = grad.data<float>();
    std::fill(grad_data, grad_data + grad.size(), 0.0F);

    const auto* feat_data = feats.data<float>();
    const auto* cotangent_data = cotangent.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();

    for (int edge = 0; edge < shape.edge_count; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        const auto kernel = kernel_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows || kernel < 0 ||
            kernel >= shape.n_kernels) {
            throw std::out_of_range(
                "spmm_edges weight gradient row index is out of bounds."
            );
        }

        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
        const auto* cotangent_row =
            cotangent_data +
            static_cast<std::ptrdiff_t>(out_row) * shape.out_channels;
        auto* grad_kernel = grad_data + static_cast<std::ptrdiff_t>(kernel) *
                                            shape.in_channels *
                                            shape.out_channels;

        for (int ci = 0; ci < shape.in_channels; ++ci) {
            auto* grad_row = grad_kernel + static_cast<std::ptrdiff_t>(ci) *
                                               shape.out_channels;
            for (int co = 0; co < shape.out_channels; ++co) {
                grad_row[co] += feat_row[ci] * cotangent_row[co];
            }
        }
    }
}

void eval_pool_edges(
    PoolReduceOp op,
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& feats = inputs[0];
    const auto& in_rows = inputs[1];
    const auto& out_rows = inputs[2];

    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto out_data = out.data<float>();
    if (op == PoolReduceOp::Sum) {
        std::fill(out_data, out_data + out.size(), 0.0F);
    } else {
        std::fill(
            out_data,
            out_data + out.size(),
            -std::numeric_limits<float>::infinity()
        );
    }

    const auto* feat_data = feats.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    for (int edge = 0; edge < shape.edge_count; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows) {
            throw std::out_of_range(
                "pool edge reduction row index is out of bounds."
            );
        }

        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        for (int channel = 0; channel < shape.channels; ++channel) {
            if (op == PoolReduceOp::Sum) {
                out_row_data[channel] += feat_row[channel];
            } else {
                out_row_data[channel] =
                    std::max(out_row_data[channel], feat_row[channel]);
            }
        }
    }
}

void eval_pool_edges_grad(
    PoolReduceOp op,
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& cotangent = inputs[0];
    const auto& feats = inputs[1];
    const auto& pooled = inputs[2];
    const auto& in_rows = inputs[3];
    const auto& out_rows = inputs[4];

    auto& grad = outputs[0];
    grad.set_data(mx::allocator::malloc(grad.nbytes()));
    auto grad_data = grad.data<float>();
    std::fill(grad_data, grad_data + grad.size(), 0.0F);

    const auto* cotangent_data = cotangent.data<float>();
    const auto* feat_data = feats.data<float>();
    const auto* pooled_data = pooled.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();

    for (int edge = 0; edge < shape.edge_count; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows) {
            throw std::out_of_range(
                "pool edge gradient row index is out of bounds."
            );
        }

        const auto* cotangent_row =
            cotangent_data +
            static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* pooled_row =
            pooled_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        auto* grad_row =
            grad_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;

        for (int channel = 0; channel < shape.channels; ++channel) {
            if (op == PoolReduceOp::Sum ||
                feat_row[channel] == pooled_row[channel]) {
                grad_row[channel] += cotangent_row[channel];
            }
        }
    }
}

void eval_pool_max_edges_jvp(
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& tangent = inputs[0];
    const auto& feats = inputs[1];
    const auto& pooled = inputs[2];
    const auto& in_rows = inputs[3];
    const auto& out_rows = inputs[4];

    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);

    const auto* tangent_data = tangent.data<float>();
    const auto* feat_data = feats.data<float>();
    const auto* pooled_data = pooled.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();

    for (int edge = 0; edge < shape.edge_count; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows) {
            throw std::out_of_range("pool max JVP row index is out of bounds.");
        }

        const auto* tangent_row =
            tangent_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* pooled_row =
            pooled_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;

        for (int channel = 0; channel < shape.channels; ++channel) {
            if (feat_row[channel] == pooled_row[channel]) {
                out_row_data[channel] += tangent_row[channel];
            }
        }
    }
}

} // namespace mlx_lattice::exec::cpu

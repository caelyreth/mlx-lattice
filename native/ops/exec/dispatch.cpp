#include "ops/exec/dispatch.h"

#include <memory>
#include <vector>

#include "backends/cpu/exec/algorithms.h"
#include "backends/metal/exec/runtime.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mlx_lattice {

mx::array dispatch_spmm_edges_input_grad(
    const mx::array& cotangent,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    SpmmEdgesShape shape
);

mx::array dispatch_spmm_edges_weight_grad(
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    SpmmEdgesShape shape
);

mx::array dispatch_pool_edges_grad(
    PoolReduceOp op,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    PoolEdgesShape shape
);

mx::array dispatch_pool_max_edges_jvp(
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    PoolEdgesShape shape
);

namespace {

mx::Device device_for(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids
) {
    if (exec::metal::supports(feats, weights, in_rows, out_rows, kernel_ids)) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_pool(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows
) {
    if (exec::metal::supports_pool(feats, in_rows, out_rows)) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_spmm_input_grad(
    const mx::array& cotangent,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids
) {
    if (exec::metal::supports_spmm_input_grad(
            cotangent, weights, in_rows, out_rows, kernel_ids
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_spmm_weight_grad(
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids
) {
    if (exec::metal::supports_spmm_weight_grad(
            feats, cotangent, in_rows, out_rows, kernel_ids
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_pool_grad(
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows
) {
    if (exec::metal::supports_pool_grad(
            cotangent, feats, pooled, in_rows, out_rows
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

class SpmmEdges final : public mx::Primitive {
  public:
    SpmmEdges(mx::Stream stream, SpmmEdgesShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_spmm_edges(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_spmm_edges(shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::SpmmEdges"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        auto out = mx::zeros(
            mx::Shape{shape_.n_out_rows, shape_.out_channels},
            primals[0].dtype(),
            stream()
        );
        auto has_tangent = false;
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 0) {
                auto component = dispatch_spmm_edges(
                    tangents[index],
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    shape_.n_out_rows
                );
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            } else if (argnums[index] == 1) {
                auto component = dispatch_spmm_edges(
                    primals[0],
                    tangents[index],
                    primals[2],
                    primals[3],
                    primals[4],
                    shape_.n_out_rows
                );
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            }
        }
        return {out};
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override {
        (void)outputs;
        std::vector<mx::array> grads;
        grads.reserve(argnums.size());
        for (const auto argnum : argnums) {
            if (argnum == 0) {
                grads.push_back(dispatch_spmm_edges_input_grad(
                    cotangents[0],
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    shape_
                ));
            } else if (argnum == 1) {
                grads.push_back(dispatch_spmm_edges_weight_grad(
                    primals[0],
                    cotangents[0],
                    primals[2],
                    primals[3],
                    primals[4],
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SpmmEdges&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
               shape_.n_kernels == op.shape_.n_kernels;
    }

  private:
    SpmmEdgesShape shape_;
};

class PoolEdges final : public mx::Primitive {
  public:
    PoolEdges(mx::Stream stream, PoolReduceOp op, PoolEdgesShape shape)
        : mx::Primitive(stream), op_(op), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_pool_edges(op_, shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_pool_edges(op_, shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::PoolEdges"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] != 0) {
                continue;
            }
            if (op_ == PoolReduceOp::Sum) {
                return {dispatch_pool_edges(
                    op_,
                    tangents[index],
                    primals[1],
                    primals[2],
                    shape_.n_out_rows
                )};
            }
            auto pooled = dispatch_pool_edges(
                op_, primals[0], primals[1], primals[2], shape_.n_out_rows
            );
            return {dispatch_pool_max_edges_jvp(
                tangents[index],
                primals[0],
                pooled,
                primals[1],
                primals[2],
                shape_
            )};
        }
        return {mx::zeros(
            mx::Shape{shape_.n_out_rows, shape_.channels},
            primals[0].dtype(),
            stream()
        )};
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override {
        std::vector<mx::array> grads;
        grads.reserve(argnums.size());
        for (const auto argnum : argnums) {
            if (argnum == 0) {
                grads.push_back(dispatch_pool_edges_grad(
                    op_,
                    cotangents[0],
                    primals[0],
                    outputs[0],
                    primals[1],
                    primals[2],
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const PoolEdges&>(other);
        return op_ == op.op_ && shape_.edge_count == op.shape_.edge_count &&
               shape_.channels == op.shape_.channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows;
    }

  private:
    PoolReduceOp op_;
    PoolEdgesShape shape_;
};

class SpmmEdgesInputGrad final : public mx::Primitive {
  public:
    SpmmEdgesInputGrad(mx::Stream stream, SpmmEdgesShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_spmm_edges_input_grad(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_spmm_edges_input_grad(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SpmmEdgesInputGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SpmmEdgesInputGrad&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
               shape_.n_kernels == op.shape_.n_kernels;
    }

  private:
    SpmmEdgesShape shape_;
};

class SpmmEdgesWeightGrad final : public mx::Primitive {
  public:
    SpmmEdgesWeightGrad(mx::Stream stream, SpmmEdgesShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_spmm_edges_weight_grad(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_spmm_edges_weight_grad(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SpmmEdgesWeightGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SpmmEdgesWeightGrad&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
               shape_.n_kernels == op.shape_.n_kernels;
    }

  private:
    SpmmEdgesShape shape_;
};

class PoolEdgesGrad final : public mx::Primitive {
  public:
    PoolEdgesGrad(mx::Stream stream, PoolReduceOp op, PoolEdgesShape shape)
        : mx::Primitive(stream), op_(op), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_pool_edges_grad(op_, shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_pool_edges_grad(
            op_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::PoolEdgesGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const PoolEdgesGrad&>(other);
        return op_ == op.op_ && shape_.edge_count == op.shape_.edge_count &&
               shape_.channels == op.shape_.channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows;
    }

  private:
    PoolReduceOp op_;
    PoolEdgesShape shape_;
};

class PoolMaxEdgesJvp final : public mx::Primitive {
  public:
    PoolMaxEdgesJvp(mx::Stream stream, PoolEdgesShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_pool_max_edges_jvp(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_pool_max_edges_jvp(shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::PoolMaxEdgesJvp"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const PoolMaxEdgesJvp&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.channels == op.shape_.channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows;
    }

  private:
    PoolEdgesShape shape_;
};

} // namespace

mx::array dispatch_spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    int n_out_rows
) {
    auto shape = SpmmEdgesShape{
        in_rows.shape(0),
        feats.shape(1),
        weights.shape(2),
        feats.shape(0),
        n_out_rows,
        weights.shape(0),
    };
    auto device = device_for(feats, weights, in_rows, out_rows, kernel_ids);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<SpmmEdges>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(weights, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(kernel_ids, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{n_out_rows, weights.shape(2)}},
        {feats.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_pool_edges(
    PoolReduceOp op,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    int n_out_rows
) {
    auto shape = PoolEdgesShape{
        in_rows.shape(0),
        feats.shape(1),
        feats.shape(0),
        n_out_rows,
    };
    auto device = device_for_pool(feats, in_rows, out_rows);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<PoolEdges>(stream, op, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{n_out_rows, feats.shape(1)}},
        {feats.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_spmm_edges_input_grad(
    const mx::array& cotangent,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    SpmmEdgesShape shape
) {
    auto device = device_for_spmm_input_grad(
        cotangent, weights, in_rows, out_rows, kernel_ids
    );
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<SpmmEdgesInputGrad>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(cotangent, false, device),
        mx::contiguous(weights, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(kernel_ids, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_in_rows, shape.in_channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_spmm_edges_weight_grad(
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    SpmmEdgesShape shape
) {
    auto device = device_for_spmm_weight_grad(
        feats, cotangent, in_rows, out_rows, kernel_ids
    );
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<SpmmEdgesWeightGrad>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(cotangent, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(kernel_ids, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_kernels, shape.in_channels, shape.out_channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_pool_edges_grad(
    PoolReduceOp op,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    PoolEdgesShape shape
) {
    auto device =
        device_for_pool_grad(cotangent, feats, pooled, in_rows, out_rows);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<PoolEdgesGrad>(stream, op, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(cotangent, false, device),
        mx::contiguous(feats, false, device),
        mx::contiguous(pooled, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_in_rows, shape.channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_pool_max_edges_jvp(
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    PoolEdgesShape shape
) {
    auto device =
        device_for_pool_grad(tangent, feats, pooled, in_rows, out_rows);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<PoolMaxEdgesJvp>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(tangent, false, device),
        mx::contiguous(feats, false, device),
        mx::contiguous(pooled, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_out_rows, shape.channels}},
        {tangent.dtype()},
        primitive,
        inputs
    )[0];
}

} // namespace mlx_lattice

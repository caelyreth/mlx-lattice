#include "ops/exec/dispatch.h"

#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "backends/cpu/exec/algorithms.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mlx_lattice {

mx::array dispatch_sparse_conv_input_grad(
    SparseMapOp op,
    const mx::array& cotangent,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparseConvShape shape
);

mx::array dispatch_sparse_conv_weight_grad(
    SparseMapOp op,
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    const mx::Shape& weight_shape,
    Triple stride,
    Triple padding,
    SparseConvShape shape
);

mx::array dispatch_sparse_pool_grad(
    PoolReduceOp reduce,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparsePoolShape shape
);

mx::array dispatch_sparse_pool_jvp(
    PoolReduceOp reduce,
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparsePoolShape shape
);

namespace {

mx::Stream sparse_exec_stream() { return mx::default_stream(mx::Device::cpu); }

class CpuSparsePrimitive : public mx::Primitive {
  public:
    using mx::Primitive::Primitive;

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) final {
        (void)inputs;
        (void)outputs;
        throw std::runtime_error(
            "Sparse execution is CPU-only until a native GPU backend is "
            "implemented."
        );
    }
};

class SparseConv final : public CpuSparsePrimitive {
  public:
    SparseConv(
        mx::Stream stream,
        SparseMapOp op,
        SparseConvShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : CpuSparsePrimitive(stream), op_(op), shape_(shape), stride_(stride),
          padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparseConv"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        auto out = mx::zeros(
            mx::Shape{shape_.out_capacity, shape_.out_channels},
            primals[2].dtype(),
            stream()
        );
        auto has_tangent = false;
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 2) {
                auto component = dispatch_sparse_conv(
                                     op_,
                                     primals[0],
                                     primals[1],
                                     tangents[index],
                                     primals[3],
                                     primals[4],
                                     stride_,
                                     padding_
                )
                                     .feats;
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            } else if (argnums[index] == 3) {
                auto component = dispatch_sparse_conv(
                                     op_,
                                     primals[0],
                                     primals[1],
                                     primals[2],
                                     tangents[index],
                                     primals[4],
                                     stride_,
                                     padding_
                )
                                     .feats;
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            }
        }
        return {
            out,
            mx::zeros(
                mx::Shape{shape_.out_capacity, 4}, primals[0].dtype(), stream()
            ),
            mx::zeros(mx::Shape{2}, mx::int32, stream()),
        };
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
            if (argnum == 2) {
                grads.push_back(dispatch_sparse_conv_input_grad(
                    op_,
                    cotangents[0],
                    primals[0],
                    primals[1],
                    primals[3],
                    primals[4],
                    stride_,
                    padding_,
                    shape_
                ));
            } else if (argnum == 3) {
                grads.push_back(dispatch_sparse_conv_weight_grad(
                    op_,
                    primals[2],
                    cotangents[0],
                    primals[0],
                    primals[1],
                    primals[4],
                    primals[3].shape(),
                    stride_,
                    padding_,
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseConv)) {
            return false;
        }
        const auto& op = static_cast<const SparseConv&>(other);
        return op_ == op.op_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.weight_layout == op.shape_.weight_layout &&
               shape_.kernel_x == op.shape_.kernel_x &&
               shape_.kernel_y == op.shape_.kernel_y &&
               shape_.kernel_z == op.shape_.kernel_z;
    }

  private:
    SparseMapOp op_;
    SparseConvShape shape_;
    Triple stride_;
    Triple padding_;
};

class SparseConvInputGrad : public CpuSparsePrimitive {
  public:
    SparseConvInputGrad(
        mx::Stream stream,
        SparseMapOp op,
        SparseConvShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : CpuSparsePrimitive(stream), op_(op), shape_(shape), stride_(stride),
          padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv_input_grad(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparseConvInputGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseConvInputGrad)) {
            return false;
        }
        const auto& op = static_cast<const SparseConvInputGrad&>(other);
        return op_ == op.op_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.weight_layout == op.shape_.weight_layout &&
               shape_.kernel_x == op.shape_.kernel_x &&
               shape_.kernel_y == op.shape_.kernel_y &&
               shape_.kernel_z == op.shape_.kernel_z;
    }

  protected:
    SparseMapOp op_;
    SparseConvShape shape_;
    Triple stride_;
    Triple padding_;
};

class SparseConvWeightGrad final : public SparseConvInputGrad {
  public:
    using SparseConvInputGrad::SparseConvInputGrad;

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv_weight_grad(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::SparseConvWeightGrad";
    }
};

class SparsePool final : public CpuSparsePrimitive {
  public:
    SparsePool(
        mx::Stream stream,
        PoolReduceOp reduce,
        SparsePoolShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : CpuSparsePrimitive(stream), reduce_(reduce), shape_(shape),
          stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_pool(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparsePool"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 2) {
                auto outputs = dispatch_sparse_pool(
                    reduce_,
                    primals[0],
                    primals[1],
                    primals[2],
                    primals[3],
                    stride_,
                    padding_
                );
                auto tangent = dispatch_sparse_pool_jvp(
                    reduce_,
                    tangents[index],
                    primals[2],
                    outputs.feats,
                    primals[0],
                    primals[1],
                    primals[3],
                    stride_,
                    padding_,
                    shape_
                );
                return {
                    tangent,
                    mx::zeros(
                        mx::Shape{shape_.out_capacity, 4},
                        primals[0].dtype(),
                        stream()
                    ),
                    mx::zeros(mx::Shape{2}, mx::int32, stream()),
                };
            }
        }
        return {
            mx::zeros(
                mx::Shape{shape_.out_capacity, shape_.channels},
                primals[2].dtype(),
                stream()
            ),
            mx::zeros(
                mx::Shape{shape_.out_capacity, 4}, primals[0].dtype(), stream()
            ),
            mx::zeros(mx::Shape{2}, mx::int32, stream()),
        };
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override {
        std::vector<mx::array> grads;
        grads.reserve(argnums.size());
        for (const auto argnum : argnums) {
            if (argnum == 2) {
                grads.push_back(dispatch_sparse_pool_grad(
                    reduce_,
                    cotangents[0],
                    primals[2],
                    outputs[SparseOutFeats],
                    primals[0],
                    primals[1],
                    primals[3],
                    stride_,
                    padding_,
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparsePool)) {
            return false;
        }
        const auto& op = static_cast<const SparsePool&>(other);
        return reduce_ == op.reduce_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.channels == op.shape_.channels;
    }

  private:
    PoolReduceOp reduce_;
    SparsePoolShape shape_;
    Triple stride_;
    Triple padding_;
};

class SparsePoolGrad : public CpuSparsePrimitive {
  public:
    SparsePoolGrad(
        mx::Stream stream,
        PoolReduceOp reduce,
        SparsePoolShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : CpuSparsePrimitive(stream), reduce_(reduce), shape_(shape),
          stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_pool_grad(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparsePoolGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparsePoolGrad)) {
            return false;
        }
        const auto& op = static_cast<const SparsePoolGrad&>(other);
        return reduce_ == op.reduce_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.channels == op.shape_.channels;
    }

  protected:
    PoolReduceOp reduce_;
    SparsePoolShape shape_;
    Triple stride_;
    Triple padding_;
};

class SparsePoolJvp final : public SparsePoolGrad {
  public:
    using SparsePoolGrad::SparsePoolGrad;

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_pool_jvp(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparsePoolJvp"; }
};

} // namespace

NativeSparseTensorOutput dispatch_sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding
) {
    auto out_capacity = op == SparseMapOp::Forward
                            ? coords.shape(0)
                            : coords.shape(0) * offsets.shape(0);
    auto mapped_weight = weights.ndim() == 3;
    auto shape = SparseConvShape{
        coords.shape(0),
        out_capacity,
        offsets.shape(0),
        feats.shape(1),
        mapped_weight ? weights.shape(2) : weights.shape(0),
        mapped_weight ? 0 : 1,
        mapped_weight ? offsets.shape(0) : weights.shape(1),
        mapped_weight ? 1 : weights.shape(2),
        mapped_weight ? 1 : weights.shape(3),
    };
    auto stream = sparse_exec_stream();
    auto primitive =
        std::make_shared<SparseConv>(stream, op, shape, stride, padding);
    auto inputs = std::vector<mx::array>{
        coords,
        active_rows,
        feats,
        weights,
        offsets,
    };
    auto outputs = mx::array::make_arrays(
        {mx::Shape{out_capacity, shape.out_channels},
         mx::Shape{out_capacity, 4},
         mx::Shape{2}},
        {feats.dtype(), coords.dtype(), mx::int32},
        primitive,
        inputs
    );
    return {
        outputs[SparseOutCoords],
        outputs[SparseOutFeats],
        outputs[SparseCounts],
    };
}

NativeSparseTensorOutput dispatch_sparse_pool(
    PoolReduceOp reduce,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets,
    Triple stride,
    Triple padding
) {
    auto out_capacity = coords.shape(0);
    auto shape = SparsePoolShape{
        coords.shape(0),
        out_capacity,
        offsets.shape(0),
        feats.shape(1),
    };
    auto stream = sparse_exec_stream();
    auto primitive =
        std::make_shared<SparsePool>(stream, reduce, shape, stride, padding);
    auto inputs = std::vector<mx::array>{
        coords,
        active_rows,
        feats,
        offsets,
    };
    auto outputs = mx::array::make_arrays(
        {mx::Shape{out_capacity, feats.shape(1)},
         mx::Shape{out_capacity, 4},
         mx::Shape{2}},
        {feats.dtype(), coords.dtype(), mx::int32},
        primitive,
        inputs
    );
    return {
        outputs[SparseOutCoords],
        outputs[SparseOutFeats],
        outputs[SparseCounts],
    };
}

mx::array dispatch_sparse_conv_input_grad(
    SparseMapOp op,
    const mx::array& cotangent,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparseConvShape shape
) {
    auto stream = sparse_exec_stream();
    auto primitive = std::make_shared<SparseConvInputGrad>(
        stream, op, shape, stride, padding
    );
    auto inputs = std::vector<mx::array>{
        cotangent,
        coords,
        active_rows,
        weights,
        offsets,
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.in_capacity, shape.in_channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_sparse_conv_weight_grad(
    SparseMapOp op,
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    const mx::Shape& weight_shape,
    Triple stride,
    Triple padding,
    SparseConvShape shape
) {
    auto stream = sparse_exec_stream();
    auto primitive = std::make_shared<SparseConvWeightGrad>(
        stream, op, shape, stride, padding
    );
    auto inputs = std::vector<mx::array>{
        feats,
        cotangent,
        coords,
        active_rows,
        offsets,
    };
    return mx::array::make_arrays(
        {weight_shape}, {cotangent.dtype()}, primitive, inputs
    )[0];
}

mx::array dispatch_sparse_pool_grad(
    PoolReduceOp reduce,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparsePoolShape shape
) {
    (void)pooled;
    auto stream = sparse_exec_stream();
    auto primitive = std::make_shared<SparsePoolGrad>(
        stream, reduce, shape, stride, padding
    );
    auto inputs = std::vector<mx::array>{
        cotangent,
        feats,
        pooled,
        coords,
        active_rows,
        offsets,
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.in_capacity, shape.channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_sparse_pool_jvp(
    PoolReduceOp reduce,
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparsePoolShape shape
) {
    auto stream = sparse_exec_stream();
    auto primitive =
        std::make_shared<SparsePoolJvp>(stream, reduce, shape, stride, padding);
    auto inputs = std::vector<mx::array>{
        tangent,
        feats,
        pooled,
        coords,
        active_rows,
        offsets,
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.out_capacity, shape.channels}},
        {tangent.dtype()},
        primitive,
        inputs
    )[0];
}

} // namespace mlx_lattice

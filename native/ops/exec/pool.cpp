#include "ops/exec/factories.h"

#include <memory>
#include <typeinfo>
#include <vector>

#include "backends/cpu/exec/algorithms.h"
#include "backends/metal/exec/runtime.h"
#include "mlx/ops.h"
#include "ops/exec/primitive.h"
#include "ops/exec/streams.h"

namespace mlx_lattice {

namespace {

mx::array make_sparse_pool_grad(
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

mx::array make_sparse_pool_jvp(
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

class SparsePool final : public SparsePrimitive {
  public:
    SparsePool(
        mx::Stream stream,
        PoolReduceOp reduce,
        SparsePoolShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : SparsePrimitive(stream), reduce_(reduce), shape_(shape),
          stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_pool(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_pool(
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
                auto outputs = make_sparse_pool(
                    reduce_,
                    primals[0],
                    primals[1],
                    primals[2],
                    primals[3],
                    stride_,
                    padding_
                );
                auto tangent = make_sparse_pool_jvp(
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
                grads.push_back(make_sparse_pool_grad(
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

class SparsePoolGrad : public SparsePrimitive {
  public:
    SparsePoolGrad(
        mx::Stream stream,
        PoolReduceOp reduce,
        SparsePoolShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : SparsePrimitive(stream), reduce_(reduce), shape_(shape),
          stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_pool_grad(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_pool_grad(
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

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_pool_jvp(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparsePoolJvp"; }
};

} // namespace

NativeSparseTensorOutput make_sparse_pool(
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
    auto stream = sparse_pool_stream(coords, active_rows, feats, offsets);
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

namespace {

mx::array make_sparse_pool_grad(
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
    auto stream = sparse_pool_stream(coords, active_rows, cotangent, offsets);
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

mx::array make_sparse_pool_jvp(
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
    auto stream = sparse_pool_stream(coords, active_rows, tangent, offsets);
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

} // namespace

} // namespace mlx_lattice

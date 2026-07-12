#include "features/convolution/api.h"

#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "features/convolution/metal/runtime.h"
#include "foundation/array_utils.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "platform/cpu/schedule.h"

namespace mlx_lattice {
namespace {

mx::Device projection_device() {
    return mx::default_device() == mx::Device::gpu ? mx::Device::gpu
                                                   : mx::Device::cpu;
}

mx::array make_precise_feature_projection(
    const mx::array& feats,
    const mx::array& weights
);

struct ProjectionShape {
    int rows;
    int in_channels;
    int out_channels;
};

class PreciseFeatureProjection final : public mx::Primitive {
  public:
    PreciseFeatureProjection(mx::Stream stream, ProjectionShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::allocate_all(outputs);
        backend::schedule_cpu(
            stream(),
            inputs,
            outputs,
            [](const std::vector<mx::array>& ready,
               std::vector<mx::array>& task_outputs) {
                const auto& feats = ready[0];
                const auto& weights = ready[1];
                auto& out = task_outputs[0];
                const auto* feat_data = feats.data<float>();
                const auto* weight_data = weights.data<float>();
                auto* out_data = out.data<float>();
                for (int row = 0; row < feats.shape(0); ++row) {
                    for (int output = 0; output < weights.shape(0); ++output) {
                        auto accumulator = 0.0F;
                        for (int input = 0; input < feats.shape(1); ++input) {
                            accumulator += feat_data
                                               [row * feats.strides(0) +
                                                input * feats.strides(1)] *
                                           weight_data
                                               [output * weights.strides(0) +
                                                input * weights.strides(1)];
                        }
                        out_data
                            [row * out.strides(0) + output * out.strides(1)] =
                                accumulator;
                    }
                }
            }
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::metal::conv::eval_projection(stream(), inputs, outputs);
    }

    const char* name() const override {
        return "lattice::PreciseFeatureProjection";
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        auto tangent = mx::zeros(
            mx::Shape{shape_.rows, shape_.out_channels}, mx::float32, stream()
        );
        auto has_tangent = false;
        for (int index = 0; index < static_cast<int>(argnums.size()); ++index) {
            if (argnums[index] == 0) {
                auto component = make_precise_feature_projection(
                    tangents[index], primals[1]
                );
                tangent = has_tangent ? mx::add(tangent, component, stream())
                                      : component;
            } else if (argnums[index] == 1) {
                auto component = make_precise_feature_projection(
                    primals[0], tangents[index]
                );
                tangent = has_tangent ? mx::add(tangent, component, stream())
                                      : component;
            } else {
                continue;
            }
            has_tangent = true;
        }
        return {tangent};
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
                grads.push_back(make_precise_feature_projection(
                    cotangents[0], mx::transpose(primals[1], stream())
                ));
            } else if (argnum == 1) {
                grads.push_back(make_precise_feature_projection(
                    mx::transpose(cotangents[0], stream()),
                    mx::transpose(primals[0], stream())
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(PreciseFeatureProjection)) {
            return false;
        }
        const auto& op = static_cast<const PreciseFeatureProjection&>(other);
        return shape_.rows == op.shape_.rows &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels;
    }

  private:
    ProjectionShape shape_;
};

mx::array make_precise_feature_projection(
    const mx::array& feats,
    const mx::array& weights
) {
    auto device = projection_device();
    return mx::array::make_arrays(
        {mx::Shape{feats.shape(0), weights.shape(0)}},
        {mx::float32},
        std::make_shared<PreciseFeatureProjection>(
            mx::default_stream(device),
            ProjectionShape{
                feats.shape(0),
                feats.shape(1),
                weights.shape(0),
            }
        ),
        {mx::contiguous(feats, false, device),
         mx::contiguous(weights, false, device)}
    )[0];
}

} // namespace

mx::array
precise_feature_projection(const mx::array& feats, const mx::array& weights) {
    if (feats.ndim() != 2 || weights.ndim() != 2) {
        throw std::invalid_argument(
            "precise_feature_projection requires rank-2 feature and weight "
            "arrays."
        );
    }
    if (feats.dtype() != mx::float32 || weights.dtype() != mx::float32) {
        throw std::invalid_argument(
            "precise_feature_projection requires float32 features and weights."
        );
    }
    if (feats.shape(1) != weights.shape(1)) {
        throw std::invalid_argument(
            "feature channels must match the weight input channels."
        );
    }
    return make_precise_feature_projection(feats, weights);
}

} // namespace mlx_lattice

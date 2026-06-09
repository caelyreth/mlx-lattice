#include "ops/coords/factories.h"

#include <cstdint>
#include <memory>
#include <typeinfo>
#include <vector>

#include "backends/cpu/coords/algorithms.h"
#include "backends/metal/coords/runtime.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "ops/coords/streams.h"

namespace mlx_lattice {

namespace {

struct RelationOutputSpec {
    std::vector<mx::Shape> shapes;
    std::vector<mx::Dtype> dtypes;
};

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

NativeKernelRelation
relation_from_outputs(const std::vector<mx::array>& outputs) {
    return {
        outputs[RelationInRows],
        outputs[RelationOutRows],
        outputs[RelationKernelIds],
        outputs[RelationOutCoords],
        outputs[RelationCounts],
    };
}

RelationOutputSpec relation_output_spec(
    int edges,    // NOLINT(bugprone-easily-swappable-parameters)
    int out_rows, // NOLINT(bugprone-easily-swappable-parameters)
    mx::Dtype coord_dtype
) {
    std::vector<mx::Shape> shapes(RelationOutputCount, mx::Shape{edges});
    std::vector<mx::Dtype> dtypes(RelationOutputCount, mx::int32);
    shapes[RelationOutCoords] = mx::Shape{out_rows, 4};
    shapes[RelationCounts] = mx::Shape{2};
    dtypes[RelationOutCoords] = coord_dtype;
    return {shapes, dtypes};
}

std::vector<mx::array> make_relation_outputs(
    const RelationOutputSpec& spec,
    const std::shared_ptr<mx::Primitive>& primitive,
    const mx::array& coords,
    const mx::array& offsets,
    const mx::array& active_rows,
    mx::Device device
) {
    auto inputs = std::vector<mx::array>{
        mx::contiguous(coords, false, device),
        mx::contiguous(offsets, false, device),
        mx::contiguous(active_rows, false, device),
    };
    return mx::array::make_arrays(spec.shapes, spec.dtypes, primitive, inputs);
}

class GenericKernelRelation final : public mx::Primitive {
  public:
    GenericKernelRelation(
        mx::Stream stream,
        CoordRelationOp op,
        int rows, // NOLINT(bugprone-easily-swappable-parameters)
        int kernel_count,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : mx::Primitive(stream), op_(op), rows_(rows),
          kernel_count_(kernel_count), stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_generic_kernel_relation(
            op_, stride_, padding_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_generic_kernel_relation(
            op_,
            rows_,
            kernel_count_,
            stride_,
            padding_,
            stream(),
            inputs,
            outputs
        );
    }

    const char* name() const override {
        return "lattice::GenericKernelRelation";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(GenericKernelRelation)) {
            return false;
        }
        const auto& relation = static_cast<const GenericKernelRelation&>(other);
        return op_ == relation.op_ && rows_ == relation.rows_ &&
               kernel_count_ == relation.kernel_count_ &&
               stride_ == relation.stride_ && padding_ == relation.padding_;
    }

  private:
    CoordRelationOp op_;
    int rows_;
    int kernel_count_;
    Triple stride_;
    Triple padding_;
};

class GenerativeKernelRelation final : public mx::Primitive {
  public:
    GenerativeKernelRelation(
        mx::Stream stream,
        int rows, // NOLINT(bugprone-easily-swappable-parameters)
        int kernel_count,
        Triple stride
    )
        : mx::Primitive(stream), rows_(rows), kernel_count_(kernel_count),
          stride_(stride) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_generative_kernel_relation(
            stride_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_generative_kernel_relation(
            rows_, kernel_count_, stride_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::GenerativeKernelRelation";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(GenerativeKernelRelation)) {
            return false;
        }
        const auto& relation =
            static_cast<const GenerativeKernelRelation&>(other);
        return rows_ == relation.rows_ &&
               kernel_count_ == relation.kernel_count_ &&
               stride_ == relation.stride_;
    }

  private:
    int rows_;
    int kernel_count_;
    Triple stride_;
};

} // namespace

NativeKernelRelation make_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    auto offsets = kernel_offsets(kernel_size, dilation);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto max_out_rows = rows;
    auto max_edges = max_out_rows * kernel_count;
    auto device = coord_device();
    auto outputs = make_relation_outputs(
        relation_output_spec(max_edges, max_out_rows, coords.dtype()),
        std::make_shared<GenericKernelRelation>(
            coord_stream(device),
            CoordRelationOp::Forward,
            rows,
            kernel_count,
            stride,
            padding
        ),
        coords,
        offset_values,
        active_rows,
        device
    );
    return relation_from_outputs(outputs);
}

NativeKernelRelation make_generative_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride
) {
    auto offsets = kernel_offsets(kernel_size);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto pair_count = rows * kernel_count;
    auto device = coord_device();
    auto outputs = make_relation_outputs(
        relation_output_spec(pair_count, pair_count, coords.dtype()),
        std::make_shared<GenerativeKernelRelation>(
            coord_stream(device), rows, kernel_count, stride
        ),
        coords,
        offset_values,
        active_rows,
        device
    );
    return relation_from_outputs(outputs);
}

NativeKernelRelation make_transposed_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    auto offsets = kernel_offsets(kernel_size, dilation);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto max_edges = rows * kernel_count;
    auto device = coord_device();
    auto outputs = make_relation_outputs(
        relation_output_spec(max_edges, max_edges, coords.dtype()),
        std::make_shared<GenericKernelRelation>(
            coord_stream(device),
            CoordRelationOp::Transposed,
            rows,
            kernel_count,
            stride,
            padding
        ),
        coords,
        offset_values,
        active_rows,
        device
    );
    return relation_from_outputs(outputs);
}

} // namespace mlx_lattice

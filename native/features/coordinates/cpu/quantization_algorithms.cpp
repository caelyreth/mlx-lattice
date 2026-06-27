#include "features/coordinates/cpu/algorithm_details.h"

namespace mlx_lattice::coords::cpu {
void eval_sparse_quantize(
    QuantizationSpec spec,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [spec](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_sparse_quantization(
                task_outputs,
                QuantizationInputs{
                    task_inputs[0],
                    task_inputs[1],
                    read_scalar_i32(task_inputs[2]),
                },
                spec
            );
        }
    );
}

void eval_voxelize_features(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_voxel_features(
                task_outputs[0],
                reduce,
                VoxelFeatureInputs{
                    task_inputs[0],
                    task_inputs[1],
                    task_inputs[2],
                    task_inputs[3],
                },
                shape
            );
        }
    );
}

void eval_voxelize_feature_grad(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_voxel_feature_grad(
                task_outputs[0],
                reduce,
                VoxelFeatureInputs{
                    task_inputs[0],
                    task_inputs[1],
                    task_inputs[2],
                    task_inputs[3],
                },
                shape
            );
        }
    );
}

void eval_point_voxel_map(
    QuantizationSpec spec,
    PointVoxelInterpolationOp interpolation,
    PointVoxelMapShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [spec, interpolation, shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_point_voxel_map(
                task_outputs,
                PointVoxelMapInputs{
                    task_inputs[0],
                    task_inputs[1],
                    task_inputs[2],
                    task_inputs[3],
                    task_inputs[4],
                },
                spec,
                interpolation,
                shape
            );
        }
    );
}

void eval_interpolate_point_features(
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_point_features(
                task_outputs[0],
                PointFeatureInputs{
                    task_inputs[0],
                    task_inputs[1],
                    task_inputs[2],
                },
                shape
            );
        }
    );
}

void eval_interpolate_point_feature_grad(
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_point_feature_grad(
                task_outputs[0],
                PointFeatureInputs{
                    task_inputs[0],
                    task_inputs[1],
                    task_inputs[2],
                },
                shape
            );
        }
    );
}

} // namespace mlx_lattice::coords::cpu

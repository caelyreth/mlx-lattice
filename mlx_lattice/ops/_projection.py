from __future__ import annotations

import mlx.core as mx

_PRECISE_SMALL_FP32_WORK = 8192


def feature_projection(x: mx.array, weight: mx.array) -> mx.array:
    """Project row-major features by an ``(C_out, C_in)`` weight matrix.

    MLX Metal's regular fp32 matmul path may use reduced-precision hardware for
    small matrices. Sparse artifact replay needs pointwise convolutions and
    linear heads to stay numerically close to CUDA/Torch fp32 reference output,
    so small fp32 projections use an explicit multiply-reduce formulation.
    Larger projections keep the regular matmul path.
    """

    if should_use_precise_small_fp32_projection(
        x,
        in_channels=int(weight.shape[1]),
        out_channels=int(weight.shape[0]),
        weight_dtype=weight.dtype,
    ):
        return mx.sum(x[:, None, :] * weight[None, :, :], axis=2)
    return x @ weight.T


def should_use_precise_small_fp32_projection(
    x: mx.array,
    *,
    in_channels: int,
    out_channels: int,
    weight_dtype: mx.Dtype,
) -> bool:
    if x.dtype != mx.float32 or weight_dtype != mx.float32:
        return False
    if mx.default_device() != mx.gpu:
        return False
    if x.ndim != 2:
        return False
    work = int(x.shape[0]) * int(in_channels) * int(out_channels)
    return work <= _PRECISE_SMALL_FP32_WORK

from __future__ import annotations

import mlx.core as mx

from mlx_lattice import _ext


def feature_projection(x: mx.array, weight: mx.array) -> mx.array:
    """Project row-major features by an ``(C_out, C_in)`` weight matrix.

    MLX Metal's regular fp32 matmul path may use reduced-precision hardware.
    Lattice instead dispatches a native FP32 primitive on Metal. Its
    implementation uses TensorOps where the shape is supported and a native
    scalar FP32 kernel otherwise; both paths are checked against the shared
    binary64 oracle.
    """

    if should_use_precise_fp32_projection(x, weight_dtype=weight.dtype):
        return _ext.precise_feature_projection(x, weight)
    return x @ weight.T


def should_use_precise_fp32_projection(
    x: mx.array,
    *,
    weight_dtype: mx.Dtype,
) -> bool:
    if x.dtype != mx.float32 or weight_dtype != mx.float32:
        return False
    if mx.default_device() != mx.gpu:
        return False
    return x.ndim == 2

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as mxnn
from lattice_contract import (
    FEATURE_BATCH_NORM,
    FEATURE_DROPOUT,
    FEATURE_GELU,
    FEATURE_LAYER_NORM,
    FEATURE_LEAKY_RELU,
    FEATURE_LINEAR,
    FEATURE_RELU,
    FEATURE_RMS_NORM,
    FEATURE_SIGMOID,
    FEATURE_SILU,
    FEATURE_SOFTPLUS,
    FEATURE_TANH,
)

from mlx_lattice.core import SparseTensor
from mlx_lattice.nn._artifact import (
    computed_attribute,
    lattice_module,
    module_parameter,
    path_attribute,
)
from mlx_lattice.ops import feature as F
from mlx_lattice.ops.feature import GeluApprox

__all__ = [
    'GELU',
    'BatchNorm',
    'Dropout',
    'LayerNorm',
    'LeakyReLU',
    'Linear',
    'RMSNorm',
    'ReLU',
    'SiLU',
    'Sigmoid',
    'Softplus',
    'Tanh',
]

if TYPE_CHECKING:
    from mlx_lattice.nn.quantized_feature import QuantizedLinear


@lattice_module(FEATURE_LINEAR, parameters=('weight', 'bias'))
class Linear(mxnn.Linear):
    """Sparse-feature linear projection module.

    This is the sparse analogue of ``mlx.nn.Linear``. It applies the dense
    projection to ``x.feats`` and preserves sparse coordinate identity.
    """

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.linear(
            x,
            self.weight,
            self.bias if 'bias' in self else None,
        )

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = 'affine',
        quantize_input: bool = False,
    ) -> QuantizedLinear:
        from mlx_lattice.nn.quantized_feature import QuantizedLinear

        if quantize_input:
            raise ValueError(
                'affine sparse QuantizedLinear uses floating-point activations.'
            )
        return QuantizedLinear.from_linear(
            self,
            group_size=group_size,
            bits=4 if bits is None else bits,
            mode=mode,
        )


@lattice_module(FEATURE_RELU)
class ReLU(mxnn.ReLU):
    """Sparse-feature ReLU module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.relu(x)


@lattice_module(FEATURE_SIGMOID)
class Sigmoid(mxnn.Sigmoid):
    """Sparse-feature sigmoid module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.sigmoid(x)


@lattice_module(
    FEATURE_GELU,
    attributes=(path_attribute('approximate', '_approx'),),
)
class GELU(mxnn.GELU):
    """Sparse-feature GELU module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.gelu(x, approximate=cast('GeluApprox', self._approx))


@lattice_module(FEATURE_SILU)
class SiLU(mxnn.SiLU):
    """Sparse-feature SiLU module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.silu(x)


@lattice_module(
    FEATURE_LEAKY_RELU,
    attributes=(path_attribute('negative_slope', '_negative_slope'),),
)
class LeakyReLU(mxnn.LeakyReLU):
    """Sparse-feature leaky ReLU module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.leaky_relu(x, negative_slope=self._negative_slope)


@lattice_module(FEATURE_TANH)
class Tanh(mxnn.Tanh):
    """Sparse-feature tanh module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.tanh(x)


@lattice_module(FEATURE_SOFTPLUS)
class Softplus(mxnn.Softplus):
    """Sparse-feature softplus module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.softplus(x)


@lattice_module(
    FEATURE_DROPOUT,
    attributes=(
        computed_attribute('p', lambda module: 1 - module._p_1),
        path_attribute('training', 'training'),
    ),
)
class Dropout(mxnn.Dropout):
    """Sparse-feature dropout module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.dropout(x, p=1 - self._p_1, training=self.training)


@lattice_module(
    FEATURE_BATCH_NORM,
    parameters=(
        'weight',
        'bias',
        module_parameter('mean', 'running_mean'),
        module_parameter('var', 'running_var'),
    ),
    attributes=(path_attribute('eps', 'eps'),),
)
class BatchNorm(mxnn.BatchNorm):
    """Sparse-feature batch normalization module.

    Statistics are computed over sparse feature rows. Running-stat behavior
    follows the underlying ``mlx.nn.BatchNorm`` fields.
    """

    def __call__(self, x: SparseTensor) -> SparseTensor:
        mean = mx.mean(x.feats, axis=0)
        var = mx.var(x.feats, axis=0)
        if self.training and self.track_running_stats:
            mu = self.momentum
            self.running_mean = (1 - mu) * self.running_mean + mu * mean
            self.running_var = (1 - mu) * self.running_var + mu * var
        elif self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        return F.batch_norm(
            x,
            weight=self.weight if 'weight' in self else None,
            bias=self.bias if 'bias' in self else None,
            mean=mean,
            var=var,
            eps=self.eps,
        )


@lattice_module(
    FEATURE_LAYER_NORM,
    parameters=('weight', 'bias'),
    attributes=(path_attribute('eps', 'eps'),),
)
class LayerNorm(mxnn.LayerNorm):
    """Sparse-feature layer normalization module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.layer_norm(
            x,
            weight=self.weight if 'weight' in self else None,
            bias=self.bias if 'bias' in self else None,
            eps=self.eps,
        )


@lattice_module(
    FEATURE_RMS_NORM,
    parameters=('weight',),
    attributes=(path_attribute('eps', 'eps'),),
)
class RMSNorm(mxnn.RMSNorm):
    """Sparse-feature RMS normalization module preserving sparse coordinates."""

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.rms_norm(x, weight=self.weight, eps=self.eps)

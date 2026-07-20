from __future__ import annotations

import pytest
from lattice_contract import submanifold_conv3d_f32_to_f64

from mlx_lattice import SparseTensor
from mlx_lattice.core import dequantize_weight, quantize_weight
from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    normalized_conv_transpose3d,
    normalized_generative_conv_transpose3d,
    normalized_subm_conv3d,
    subm_conv3d,
)
from mlx_lattice.ops._relation_exec import (
    sparse_conv_features_sorted_direct_reference_from_relation,
)
from tests.support import (
    active_coords,
    active_feats,
    assert_nested_close,
    assert_same_sparse_identity,
    mx,
)

pytestmark = [
    pytest.mark.ops,
    pytest.mark.conv,
    pytest.mark.usefixtures('selected_backend'),
]


def test_conv3d_pointwise_matches_dense_linear_contract() -> None:
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.array([[2.0, 3.0], [5.0, 7.0]], dtype=mx.float32)
    bias = mx.array([1.0, -1.0], dtype=mx.float32)

    out = conv3d(x, weight, bias, kernel_size=1)

    assert out.feats.tolist() == [[9.0, 18.0], [19.0, 42.0]]
    assert_same_sparse_identity(out, x)


def test_normalized_subm_conv3d_matches_weight_norm_contract() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    x = SparseTensor(
        coords,
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
    )
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )
    bias = mx.array([0.25], dtype=mx.float32)

    out = normalized_subm_conv3d(x, weight, bias, kernel_size=(3, 1, 1))
    expected = mx.array(
        [
            [8.0 / (13.0**0.5) + 0.25],
            [14.0 / (14.0**0.5) + 0.25],
            [8.0 / (5.0**0.5) + 0.25],
        ],
        dtype=mx.float32,
    )
    mx.eval(out.feats, expected)

    assert_same_sparse_identity(out, x)
    assert mx.allclose(out.feats, expected, rtol=1e-6, atol=1e-6).item()


def test_normalized_pointwise_convolution_bypasses_weight_norm() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0]], dtype=mx.int32),
        mx.array([[2.0, 3.0]], dtype=mx.float32),
    )
    weight = mx.array([[4.0, 5.0]], dtype=mx.float32)

    out = normalized_subm_conv3d(x, weight, kernel_size=1)

    assert out.feats.tolist() == [[23.0]]


def test_normalized_generative_transpose_reuses_generated_support() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0]], dtype=mx.int32),
        mx.array([[2.0]], dtype=mx.float32),
        stride=2,
    )
    weight = mx.array([1.0, 2.0], dtype=mx.float32).reshape(1, 2, 1, 1, 1)

    out = normalized_generative_conv_transpose3d(
        x, weight, kernel_size=(2, 1, 1), stride=2
    )
    numerator = generative_conv_transpose3d(
        x, weight, kernel_size=(2, 1, 1), stride=2
    )
    denominator = generative_conv_transpose3d(
        x.replace(feats=mx.ones_like(x.feats)),
        mx.square(weight),
        kernel_size=(2, 1, 1),
        stride=2,
    )
    expected = numerator.feats / mx.sqrt(denominator.feats + 1e-8)
    mx.eval(out.feats, expected)

    assert mx.array_equal(out.coords, numerator.coords).item()
    assert mx.allclose(out.feats, expected, rtol=1e-6, atol=1e-6).item()


def test_generative_conv_transpose3d_runs_generic_fp16_relation_on_metal(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('float16 sparse relation kernels are Metal-only')
    x = SparseTensor(
        mx.array([[0, 0, 0, 0]], dtype=mx.int32),
        mx.ones((1, 64), dtype=mx.float16),
    )
    weight = mx.ones((64, 2, 2, 2, 64), dtype=mx.float16)

    out = generative_conv_transpose3d(
        x,
        weight,
        kernel_size=2,
        stride=1,
    )
    mx.eval(out.feats)

    assert out.feats.dtype == mx.float16
    assert out.feats.shape == (8, 64)
    assert mx.allclose(
        out.feats,
        mx.full((8, 64), 64.0, dtype=mx.float16),
        rtol=1e-3,
        atol=1e-3,
    ).item()


def test_generative_conv_transpose3d_fp16_large_relation_on_metal(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('float16 sparse relation kernels are Metal-only')
    input_rows = 6_250
    axis = mx.arange(input_rows, dtype=mx.int32)
    zeros = mx.zeros((input_rows,), dtype=mx.int32)
    x = SparseTensor(
        mx.stack((zeros, axis * 2, zeros, zeros), axis=1),
        mx.ones((input_rows, 64), dtype=mx.float16),
    )
    weight = mx.ones((64, 2, 2, 2, 64), dtype=mx.float16)

    out = generative_conv_transpose3d(
        x,
        weight,
        kernel_size=2,
        stride=1,
    )
    mx.eval(out.feats)

    assert int(out.active_rows.item()) == 50_000
    assert out.feats.shape == (50_000, 64)
    assert mx.allclose(
        out.feats,
        mx.full((50_000, 64), 64.0, dtype=mx.float16),
        rtol=1e-3,
        atol=1e-3,
    ).item()


def test_conv3d_pointwise_uses_accurate_fp32_projection(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('accurate fp32 projection is Metal-specific')
    coords = mx.array([[0, row, 0, 0] for row in range(6)], dtype=mx.int32)
    feats = mx.array(
        [
            [-0.18668198585510254, -0.07669853419065475],
            [1.0137652158737183, 0.40206432342529297],
            [0.7205407619476318, -0.8937806487083435],
            [-0.4324185848236084, 0.710955023765564],
            [0.11673124879598618, 0.2444327175617218],
            [-0.6306809186935425, -0.34102120995521545],
        ],
        dtype=mx.float32,
    )
    weight = mx.array(
        [
            [-0.6301173567771912, 0.5361918807029724],
            [-0.6141055226325989, 0.4619714915752411],
            [-0.40786921977996826, 0.07362017035484314],
        ],
        dtype=mx.float32,
    ).reshape(3, 1, 1, 1, 2)
    bias = mx.array(
        [0.17754632234573364, -0.08261620253324509, -0.397258996963501],
        dtype=mx.float32,
    )
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    expected = feats @ weight[:, 0, 0, 0, :].T + bias
    mx.eval(expected)
    mx.set_default_device(previous_device)

    out = conv3d(SparseTensor(coords, feats), weight, bias, kernel_size=1)
    mx.eval(out.feats)

    assert mx.allclose(out.feats, expected, rtol=1e-7, atol=1e-7).item()


def test_conv3d_generic_matches_fused_native_reference() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    out = conv3d(x, weight, kernel_size=(3, 1, 1))

    assert active_coords(out) == coords.tolist()
    assert active_feats(out).tolist() == [[8.0], [14.0], [8.0]]
    assert out.stride == (1, 1, 1)
    assert out.coord_manager is x.coord_manager
    assert out.coord_key != x.coord_key
    assert out.coord_manager.owns(out.coord_key)


def test_subm_conv3d_matches_contract_oracle_for_non_cubic_kernel() -> None:
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 2, 0, 1],
            [0, 0, 0, 3],
            [0, 1, 0, 3],
            [0, 2, 0, 3],
        ],
        dtype=mx.int32,
    )
    feats = mx.array([[float(index)] for index in range(1, 10)])
    kernel_size = (3, 1, 5)
    rows = tuple(float(index) for index in range(1, 16))
    weight = mx.array(rows, dtype=mx.float32).reshape((1, 3, 1, 5, 1))
    expected = submanifold_conv3d_f32_to_f64(
        coords.tolist(),
        feats.tolist(),
        tuple(((value,),) for value in rows),
        kernel_size=kernel_size,
    )

    x = SparseTensor(coords, feats)
    out = subm_conv3d(x, weight, kernel_size=kernel_size)
    mx.eval(out.feats)

    assert_same_sparse_identity(out, x)
    assert active_feats(out).tolist() == [list(row) for row in expected]


def test_subm_conv3d_fp32_stays_within_contract_oracle_tolerance() -> None:
    kernel_size = (3, 3, 3)
    in_channels, out_channels = 7, 5
    coords = mx.array(
        [
            [0, x, y, z]
            for x in range(3)
            for y in range(3)
            for z in range(3)
        ],
        dtype=mx.int32,
    )
    feats = mx.array(
        [
            [
                (-1.0 if (row + channel) % 2 else 1.0)
                * (0.125 + ((row * 11 + channel * 7) % 53) / 53.0)
                for channel in range(in_channels)
            ]
            for row in range(int(coords.shape[0]))
        ],
        dtype=mx.float32,
    )
    rows = tuple(
        tuple(
            tuple(
                (-1.0 if (kernel + input_channel + output) % 2 else 1.0)
                * (
                    0.125
                    + ((kernel * 17 + input_channel * 5 + output) % 47)
                    / 47.0
                )
                for output in range(out_channels)
            )
            for input_channel in range(in_channels)
        )
        for kernel in range(27)
    )
    weight = mx.array(
        [
            [
                [
                    [
                        [
                            rows[(x * 3 + y) * 3 + z][input_channel][output]
                            for input_channel in range(in_channels)
                        ]
                        for z in range(3)
                    ]
                    for y in range(3)
                ]
                for x in range(3)
            ]
            for output in range(out_channels)
        ],
        dtype=mx.float32,
    )
    expected = mx.array(
        submanifold_conv3d_f32_to_f64(
            coords.tolist(),
            feats.tolist(),
            rows,
            kernel_size=kernel_size,
        ),
        dtype=mx.float32,
    )

    out = subm_conv3d(
        SparseTensor(coords, feats), weight, kernel_size=kernel_size
    )
    mx.eval(out.feats, expected)

    assert mx.allclose(out.feats, expected, rtol=2e-6, atol=1e-5).item()


@pytest.mark.parametrize(
    ('rows', 'in_channels', 'out_channels'),
    [(9, 37, 29), (128, 32, 32)],
)
def test_conv3d_pointwise_uses_accurate_fp32_projection_at_any_size(
    selected_backend,
    rows: int,
    in_channels: int,
    out_channels: int,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('accurate fp32 projection is Metal-specific')
    coords = mx.array(
        [[0, row, 0, 0] for row in range(rows)], dtype=mx.int32
    )
    features = mx.array(
        [
            [
                (-1.0 if (row + channel) % 2 else 1.0)
                * (0.125 + ((row * 31 + channel * 17) % 97) / 97.0)
                * 1.0e4
                for channel in range(in_channels)
            ]
            for row in range(rows)
        ],
        dtype=mx.float32,
    )
    weight = mx.array(
        [
            [
                (-1.0 if (output * 7 + channel * 3) % 2 else 1.0)
                * (0.125 + ((output * 19 + channel * 13) % 89) / 89.0)
                * 1.0e-4
                for channel in range(in_channels)
            ]
            for output in range(out_channels)
        ],
        dtype=mx.float32,
    )
    expected = submanifold_conv3d_f32_to_f64(
        coords.tolist(),
        features.tolist(),
        (
            tuple(
                tuple(
                    float(weight[output, channel].item())
                    for output in range(out_channels)
                )
                for channel in range(in_channels)
            ),
        ),
        kernel_size=1,
    )

    out = conv3d(SparseTensor(coords, features), weight, kernel_size=1)
    reference = mx.array(expected, dtype=mx.float32)
    mx.eval(out.feats, reference)

    assert float(mx.max(mx.abs(out.feats - reference)).item()) <= 5e-6


def test_conv3d_pointwise_native_projection_preserves_vjp_contract(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('native fp32 projection is Metal-specific')
    rows, channels = 128, 32
    coords = mx.array(
        [[0, row, 0, 0] for row in range(rows)], dtype=mx.int32
    )
    feats = mx.array(
        [
            [
                ((row * 17 + channel * 5) % 41 - 20) / 41.0
                for channel in range(channels)
            ]
            for row in range(rows)
        ],
        dtype=mx.float32,
    )
    weight = mx.array(
        [
            [
                ((output * 13 + channel * 7) % 37 - 18) / 37.0
                for channel in range(channels)
            ]
            for output in range(channels)
        ],
        dtype=mx.float32,
    )

    def loss(features: mx.array, matrix: mx.array) -> mx.array:
        return mx.sum(
            conv3d(
                SparseTensor(coords, features), matrix, kernel_size=1
            ).feats
        )

    grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(feats, weight)
    expected_feats = mx.broadcast_to(mx.sum(weight, axis=0), feats.shape)
    expected_weight = mx.broadcast_to(
        mx.sum(feats, axis=0)[None, :], weight.shape
    )
    mx.eval(grad_feats, grad_weight, expected_feats, expected_weight)

    assert mx.allclose(
        grad_feats, expected_feats, rtol=2e-6, atol=2e-6
    ).item()
    assert mx.allclose(
        grad_weight, expected_weight, rtol=2e-6, atol=2e-6
    ).item()

    def project(features: mx.array, matrix: mx.array) -> mx.array:
        return conv3d(
            SparseTensor(coords, features), matrix, kernel_size=1
        ).feats

    (_,), (feature_jvp,) = mx.jvp(
        project,
        (feats, weight),
        (mx.ones_like(feats), mx.zeros_like(weight)),
    )
    (_,), (weight_jvp,) = mx.jvp(
        project,
        (feats, weight),
        (mx.zeros_like(feats), mx.ones_like(weight)),
    )
    expected_feature_jvp = mx.broadcast_to(
        mx.sum(weight, axis=1)[None, :], feature_jvp.shape
    )
    expected_weight_jvp = mx.broadcast_to(
        mx.sum(feats, axis=1)[:, None], weight_jvp.shape
    )
    mx.eval(
        feature_jvp,
        weight_jvp,
        expected_feature_jvp,
        expected_weight_jvp,
    )

    assert mx.array_equal(feature_jvp, expected_feature_jvp).item()
    assert mx.array_equal(weight_jvp, expected_weight_jvp).item()


def test_conv3d_generic_supports_float16() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float16)
    x = SparseTensor(coords, feats)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float16).reshape(
        1, 3, 1, 1, 1
    )

    out = conv3d(x, weight, kernel_size=(3, 1, 1))
    mx.eval(out.feats)

    assert out.feats.dtype == mx.float16
    assert_nested_close(
        active_feats(out).astype(mx.float32).tolist(),
        [[8.0], [14.0], [8.0]],
    )


@pytest.mark.parametrize('bits', [4, 8])
def test_packed_quantized_convolution_matches_dequantized_contract(
    bits: int,
) -> None:
    coords = mx.array(
        [[0, index, 0, 0] for index in range(12)],
        dtype=mx.int32,
    )
    feats = mx.array(
        [
            [((row + 1) * (channel + 3) % 19) / 19 for channel in range(16)]
            for row in range(12)
        ],
        dtype=mx.float16,
    )
    weights = {
        1: mx.array(
            [((index % 29) - 14) / 29 for index in range(12 * 16)],
            dtype=mx.float16,
        ).reshape((12, 1, 1, 1, 16)),
        2: mx.array(
            [((index % 31) - 15) / 31 for index in range(12 * 2 * 16)],
            dtype=mx.float16,
        ).reshape((12, 2, 1, 1, 16)),
        3: mx.array(
            [((index % 37) - 18) / 37 for index in range(12 * 3 * 16)],
            dtype=mx.float16,
        ).reshape((12, 3, 1, 1, 16)),
    }
    quantized = {
        size: quantize_weight(weight, bits=bits)
        for size, weight in weights.items()
    }
    references = {
        size: dequantize_weight(weight)
        for size, weight in quantized.items()
    }

    x = SparseTensor(coords, feats)
    transposed = SparseTensor(coords, feats, stride=2)
    outputs = (
        (
            conv3d(x, quantized[1], kernel_size=1),
            conv3d(x, references[1], kernel_size=1),
        ),
        (
            conv3d(x, quantized[3], kernel_size=(3, 1, 1)),
            conv3d(x, references[3], kernel_size=(3, 1, 1)),
        ),
        (
            subm_conv3d(x, quantized[3], kernel_size=(3, 1, 1)),
            subm_conv3d(x, references[3], kernel_size=(3, 1, 1)),
        ),
        (
            conv3d(
                x,
                quantized[3],
                kernel_size=(3, 1, 1),
                coordinates=x,
            ),
            conv3d(
                x,
                references[3],
                kernel_size=(3, 1, 1),
                coordinates=x,
            ),
        ),
        (
            conv_transpose3d(
                transposed,
                quantized[2],
                kernel_size=(2, 1, 1),
                stride=2,
            ),
            conv_transpose3d(
                transposed,
                references[2],
                kernel_size=(2, 1, 1),
                stride=2,
            ),
        ),
        (
            generative_conv_transpose3d(
                transposed,
                quantized[2],
                kernel_size=(2, 1, 1),
                stride=2,
            ),
            generative_conv_transpose3d(
                transposed,
                references[2],
                kernel_size=(2, 1, 1),
                stride=2,
            ),
        ),
    )
    mx.eval(*(item.feats for pair in outputs for item in pair))

    for actual, expected in outputs:
        assert bool(
            mx.allclose(actual.feats, expected.feats, rtol=2e-2, atol=4e-3)
        )

    packed = quantized[3]
    assert packed.weight.dtype == mx.uint32
    assert packed.in_channels == 16
    assert packed.storage_in_channels == 32
    memory_weight = mx.ones((64, 3, 3, 3, 64), dtype=mx.float16)
    memory_packed = quantize_weight(memory_weight, bits=bits)
    mx.eval(
        memory_packed.weight, memory_packed.scales, memory_packed.biases
    )
    assert memory_packed.nbytes < memory_weight.nbytes


@pytest.mark.parametrize('bits', [4, 8])
def test_sorted_quantized_implicit_gemm_matches_dequantized_contract(
    bits: int,
) -> None:
    rows = 96
    in_channels = 64
    out_channels = 32
    coords = mx.array(
        [[0, index, 0, 0] for index in range(rows)],
        dtype=mx.int32,
    )
    feats = mx.array(
        [
            [
                ((row + 1) * (channel + 3) % 23) / 23
                for channel in range(in_channels)
            ]
            for row in range(rows)
        ],
        dtype=mx.float16,
    )
    weight = mx.array(
        [
            ((index % 31) - 15) / 31
            for index in range(out_channels * 27 * in_channels)
        ],
        dtype=mx.float16,
    ).reshape((out_channels, 3, 3, 3, in_channels))
    packed = quantize_weight(weight, bits=bits)
    reference_weight = dequantize_weight(packed)
    x = SparseTensor(coords, feats)

    actual = conv3d(x, packed, kernel_size=3)
    expected = conv3d(x, reference_weight, kernel_size=3)
    mx.eval(actual.feats, expected.feats)

    assert bool(
        mx.allclose(actual.feats, expected.feats, rtol=3e-2, atol=2e-2)
    )


def test_sorted_implicit_gemm_direct_reference_matches_classic(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('sorted direct row-stationary reference is Metal-only')
    coords = mx.array(
        [[0, index, 0, 0] for index in range(96)],
        dtype=mx.int32,
    )
    feats = mx.array(
        [
            [
                ((row + 1) * (channel + 3) % 17) / 17.0
                for channel in range(32)
            ]
            for row in range(96)
        ],
        dtype=mx.float16,
    )
    x = SparseTensor(coords, feats)
    weight = mx.array(
        [((index % 23) - 11) / 23.0 for index in range(32 * 27 * 32)],
        dtype=mx.float16,
    ).reshape((32, 3, 3, 3, 32))
    relation = x.coord_manager.kernel_relation(x.coord_key, kernel_size=3)
    automatic = conv3d(x, weight, kernel_size=3).feats
    direct = sparse_conv_features_sorted_direct_reference_from_relation(
        x.feats,
        weight,
        relation,
        store_sorted=True,
    )
    reorder_rows = relation.require_sorted_implicit_gemm().reorder_rows
    mx.eval(automatic, direct, reorder_rows)

    sorted_automatic = automatic[reorder_rows].astype(mx.float32)
    assert float(mx.max(mx.abs(sorted_automatic - direct)).item()) <= 0.003


def test_sorted_implicit_gemm_preserves_convolution_autodiff_contract(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('sorted implicit GEMM is Metal-only')
    coords = mx.array(
        [[0, index, 0, 0] for index in range(96)],
        dtype=mx.int32,
    )
    feats = mx.ones((96, 32), dtype=mx.float16)
    weight = mx.ones((32, 3, 3, 3, 32), dtype=mx.float16) / 32

    def loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(conv3d(x, weight_arg, kernel_size=3).feats)

    gradients = mx.grad(loss, argnums=(0, 1))(feats, weight)
    mx.eval(*gradients)

    assert gradients[0].shape == feats.shape
    assert gradients[1].shape == weight.shape
    assert gradients[0].dtype == mx.float16
    assert gradients[1].dtype == mx.float16


def test_conv3d_automatic_dispatch_falls_back_for_unsupported_igemm_shape(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('automatic Metal dispatch policy is Metal-only')
    coords = mx.array(
        [[0, index, 0, 0] for index in range(32)],
        dtype=mx.int32,
    )
    feats = mx.ones((32, 16), dtype=mx.float16)
    weight = mx.ones((16, 3, 3, 3, 16), dtype=mx.float16) / 16
    x = SparseTensor(coords, feats)

    out = conv3d(x, weight, kernel_size=3)
    mx.eval(out.feats)

    assert out.feats.shape == (32, 16)
    assert out.feats.dtype == mx.float16


def test_conv3d_target_coordinates_match_sparse_reference() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    target_tensor = SparseTensor(
        target,
        mx.zeros((2, 1), dtype=mx.float32),
        coord_manager=x.coord_manager,
    )
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    out = conv3d(
        x,
        weight,
        kernel_size=(3, 1, 1),
        coordinates=target_tensor,
    )

    assert active_coords(out) == target.tolist()
    assert active_feats(out).tolist() == [[14.0], [3.0]]
    assert out.stride == x.stride
    assert out.coord_manager is x.coord_manager
    assert out.coord_key == target_tensor.coord_key


def test_conv3d_target_same_reuses_input_coordinate_identity() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    target = SparseTensor(
        coords,
        mx.zeros_like(feats),
        coord_manager=x.coord_manager,
    )
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    out = conv3d(
        x,
        weight,
        kernel_size=(3, 1, 1),
        coordinates=target,
    )

    assert target.coord_key == x.coord_key
    assert out.coord_key == x.coord_key
    assert active_feats(out).tolist() == [[8.0], [14.0], [8.0]]


def test_conv3d_pointwise_target_coordinates_use_sparse_relation() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.array([[2.0]], dtype=mx.float32)

    out = conv3d(x, weight, kernel_size=1, coordinates=target)

    assert active_coords(out) == target.tolist()
    assert active_feats(out).tolist() == [[4.0], [0.0]]


def test_conv3d_target_path_is_autogradable() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )

    def loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(
            conv3d(
                x,
                weight_arg,
                kernel_size=(3, 1, 1),
                coordinates=target,
            ).feats
        )

    grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(feats, weight)

    assert grad_feats.tolist() == [[1.0], [2.0], [4.0]]
    assert grad_weight.tolist() == [[[[[4.0]]], [[[2.0]]], [[[3.0]]]]]


def test_conv_transpose3d_target_coordinates_match_indexed_geometry() -> (
    None
):
    source = SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
        stride=(2, 1, 1),
    )
    target = mx.array(
        [[0, index, 0, 0] for index in range(5)], dtype=mx.int32
    )
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    out = conv_transpose3d(
        source,
        weight,
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
        padding=(1, 0, 0),
        coordinates=target,
    )

    assert active_coords(out) == target.tolist()
    assert active_feats(out).tolist() == [[2.0], [5.0], [4.0], [9.0], [6.0]]
    assert out.stride == (1, 1, 1)


def test_normalized_conv_transpose3d_target_matches_weight_norm() -> None:
    source = SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[2.0], [4.0]], dtype=mx.float32),
        stride=(2, 1, 1),
    )
    target = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    numerator = conv_transpose3d(
        source,
        weight,
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
        padding=(1, 0, 0),
        coordinates=target,
    )
    denominator = conv_transpose3d(
        source.replace(feats=mx.ones_like(source.feats)),
        mx.square(weight),
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
        padding=(1, 0, 0),
        coordinates=target,
    )
    actual = normalized_conv_transpose3d(
        source,
        weight,
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
        padding=(1, 0, 0),
        coordinates=target,
    )
    expected = numerator.feats / mx.sqrt(denominator.feats + 1e-8)
    mx.eval(actual.feats, expected)

    assert active_coords(actual) == target.tolist()
    assert_nested_close(actual.feats.tolist(), expected.tolist())


def test_conv3d_generic_path_is_autogradable_for_features_and_weights() -> (
    None
):
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )

    def loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats)

    grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(feats, weight)

    assert grad_feats.tolist() == [[3.0], [6.0], [5.0]]
    assert grad_weight.tolist() == [[[[[3.0]]], [[[6.0]]], [[[5.0]]]]]


def test_convolution_modes_are_autogradable_for_features_and_weights() -> (
    None
):
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )

    def subm_loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(
            subm_conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats
        )

    subm_grad_feats, subm_grad_weight = mx.grad(
        subm_loss,
        argnums=(0, 1),
    )(feats, weight)

    assert subm_grad_feats.tolist() == [[3.0], [6.0], [5.0]]
    assert subm_grad_weight.tolist() == [[[[[3.0]]], [[[6.0]]], [[[5.0]]]]]

    transpose_coords = mx.array([[0, 1, 0, 0]], dtype=mx.int32)
    transpose_feats = mx.array([[4.0]], dtype=mx.float32)
    transpose_weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(
        1,
        2,
        1,
        1,
        1,
    )

    def transpose_loss(
        feats_arg: mx.array,
        weight_arg: mx.array,
    ) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return mx.sum(
            conv_transpose3d(
                x,
                weight_arg,
                kernel_size=(2, 1, 1),
                stride=(2, 1, 1),
            ).feats
        )

    def generative_loss(
        feats_arg: mx.array,
        weight_arg: mx.array,
    ) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return mx.sum(
            generative_conv_transpose3d(
                x,
                weight_arg,
                kernel_size=(2, 1, 1),
                stride=(2, 1, 1),
            ).feats
        )

    expected_weight_grad = [[[[[4.0]]], [[[4.0]]]]]
    for loss in (transpose_loss, generative_loss):
        grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(
            transpose_feats,
            transpose_weight,
        )
        assert grad_feats.tolist() == [[5.0]]
        assert grad_weight.tolist() == expected_weight_grad


def test_conv3d_generic_supports_explicit_vjp_and_jvp_transforms() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    tangent = mx.ones_like(feats)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )
    weight_tangent = mx.ones_like(weight)

    def features(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats

    outputs, grads = mx.vjp(
        features,
        [feats, weight],
        [mx.ones((3, 1), dtype=mx.float32)],
    )
    _, jvps = mx.jvp(
        features,
        [feats, weight],
        [tangent, weight_tangent],
    )

    assert outputs[0].tolist() == [[8.0], [14.0], [8.0]]
    assert grads[0].tolist() == [[3.0], [6.0], [5.0]]
    assert grads[1].tolist() == [[[[[3.0]]], [[[6.0]]], [[[5.0]]]]]
    assert jvps[0].tolist() == [[8.0], [12.0], [8.0]]


def test_convolution_modes_are_compatible_with_mx_compile(
    compile_backend,
) -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

    def generic(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        return conv3d(
            SparseTensor(coords, feats_arg),
            weight_arg,
            kernel_size=(3, 1, 1),
        ).feats

    def subm(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        return subm_conv3d(
            SparseTensor(coords, feats_arg),
            weight_arg,
            kernel_size=(3, 1, 1),
        ).feats

    for fn in (generic, subm):
        compiled = mx.compile(fn)
        assert compiled(feats, weight).tolist() == [[3.0], [6.0], [5.0]]

    transpose_coords = mx.array([[0, 1, 0, 0]], dtype=mx.int32)
    transpose_feats = mx.array([[4.0]], dtype=mx.float32)
    transpose_weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(
        1,
        2,
        1,
        1,
        1,
    )

    def transposed(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return conv_transpose3d(
            x,
            weight_arg,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        ).feats

    def generated(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return generative_conv_transpose3d(
            x,
            weight_arg,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        ).feats

    for fn in (transposed, generated):
        compiled = mx.compile(fn)
        assert compiled(
            transpose_feats,
            transpose_weight,
        ).tolist() == [[8.0], [12.0]]


def test_conv3d_strided_updates_output_stride_and_coordinates() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0], [4.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((1, 1, 1, 1, 1), dtype=mx.float32)

    out = conv3d(x, weight, kernel_size=1, stride=2)

    assert active_coords(out) == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert active_feats(out).tolist() == [[1.0], [3.0]]
    assert out.stride == (2, 2, 2)


def test_subm_conv3d_reuses_input_coordinate_identity() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

    out = subm_conv3d(x, weight, kernel_size=(3, 1, 1))
    relation = x.coord_manager.submanifold_kernel_relation(
        x.coord_key,
        kernel_size=(3, 1, 1),
    )

    assert relation.contract.kind == 'submanifold'
    assert out.feats.tolist() == [[3.0], [6.0], [5.0]]
    assert_same_sparse_identity(out, x)


def test_subm_conv3d_supports_dilated_centered_kernels() -> None:
    coords = mx.array(
        [[0, x, 0, 0] for x in range(5)],
        dtype=mx.int32,
    )
    feats = mx.arange(1, 6, dtype=mx.float32).reshape(-1, 1)
    x = SparseTensor(coords, feats)
    weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

    out = subm_conv3d(
        x,
        weight,
        kernel_size=(3, 1, 1),
        dilation=(2, 1, 1),
    )

    assert out.feats.tolist() == [[4.0], [6.0], [9.0], [6.0], [8.0]]
    assert_same_sparse_identity(out, x)


def test_transpose_convs_generate_the_same_output_contract() -> None:
    x = SparseTensor(
        mx.array([[0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[4.0]], dtype=mx.float32),
        stride=(2, 1, 1),
    )
    weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(1, 2, 1, 1, 1)

    out = conv_transpose3d(
        x,
        weight,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )
    generated = generative_conv_transpose3d(
        x,
        weight,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )

    assert active_coords(out) == [[0, 2, 0, 0], [0, 3, 0, 0]]
    assert active_feats(out).tolist() == [[8.0], [12.0]]
    assert out.stride == (1, 1, 1)
    assert active_coords(generated) == active_coords(out)
    assert active_feats(generated).tolist() == active_feats(out).tolist()
    assert generated.stride == out.stride


@pytest.mark.parametrize('kernel_size', [2, 3])
@pytest.mark.parametrize('stride', [1, 2])
def test_generative_transpose_uses_every_canonical_kernel_row(
    kernel_size: int, stride: int
) -> None:
    """Generated and explicit support agree for every z-fastest kernel row."""
    source = [2, 3, 4]
    x = SparseTensor(
        mx.array([[0, *source]], dtype=mx.int32),
        mx.ones((1, 1), dtype=mx.float32),
        stride=stride,
    )
    kernel_volume = kernel_size**3

    for kernel_row in range(kernel_volume):
        values = [0.0] * kernel_volume
        values[kernel_row] = 1.0
        weight = mx.array(values, dtype=mx.float32).reshape(
            1, kernel_size, kernel_size, kernel_size, 1
        )
        generated = generative_conv_transpose3d(
            x,
            weight,
            kernel_size=kernel_size,
            stride=stride,
        )
        explicit = generative_conv_transpose3d(
            x,
            weight,
            kernel_size=kernel_size,
            stride=stride,
            coordinates=generated.coords,
        )
        offset = (
            kernel_row // (kernel_size * kernel_size),
            (kernel_row // kernel_size) % kernel_size,
            kernel_row % kernel_size,
        )
        expected_coord = [
            0,
            *(source[axis] * stride + offset[axis] for axis in range(3)),
        ]
        coords = active_coords(generated)
        features = active_feats(generated).tolist()

        assert generated.stride == (1, 1, 1)
        assert expected_coord in coords
        assert features[coords.index(expected_coord)] == [1.0]
        assert all(
            feature == [0.0]
            for coord, feature in zip(coords, features, strict=True)
            if coord != expected_coord
        )
        assert active_coords(explicit) == coords
        assert active_feats(explicit).tolist() == features


def test_conv_ops_reject_ambiguous_contracts() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0]], dtype=mx.int32),
        mx.ones((1, 1), dtype=mx.float32),
    )

    with pytest.raises(ValueError, match='odd kernel_size'):
        subm_conv3d(x, mx.ones((2, 1, 1), dtype=mx.float32), kernel_size=2)

    with pytest.raises(ValueError, match='must divide'):
        conv_transpose3d(x, mx.ones((2, 1, 1), dtype=mx.float32))

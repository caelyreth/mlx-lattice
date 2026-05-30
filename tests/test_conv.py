import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import SparseTensor, conv3d, pool3d  # noqa: E402


def assert_allclose(actual, expected, *, rtol=1e-5, atol=1e-6):
    mx.eval(actual)
    assert mx.allclose(actual, expected, rtol=rtol, atol=atol)


def test_conv3d_k3s1_identity_center_weight():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.zeros((27, 1, 1), dtype=mx.float32)
    weight = mx.concatenate(
        [weight[:13], mx.ones((1, 1, 1), dtype=mx.float32), weight[14:]],
        axis=0,
    )

    out = conv3d(x, weight, kernel_size=3, stride=1)

    assert out.coords.tolist() == coords.tolist()
    assert_allclose(out.feats, feats)


def test_conv3d_k3s1_neighbor_sum():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((27, 1, 1), dtype=mx.float32)

    out = conv3d(x, weight, kernel_size=3, stride=1)

    assert_allclose(out.feats, mx.array([[3.0], [6.0], [5.0]]))


def test_pool3d_k2s2():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = pool3d(x, kernel_size=2, stride=2)

    assert out.coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert_allclose(out.feats, mx.array([[3.0], [3.0]]))

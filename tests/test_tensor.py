import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import SparseTensor  # noqa: E402


def test_sparse_tensor_validates_shape():
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    feats = mx.ones((1, 2), dtype=mx.float32)

    x = SparseTensor(coords, feats, stride=(1, 2, 3))

    assert x.coords is coords
    assert x.feats is feats
    assert x.stride == (1, 2, 3)
    assert x.n_points == 1
    assert x.channels == 2


def test_sparse_tensor_reuses_kernel_map():
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.ones((2, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats)

    first = x.kernel_map(kernel_size=3, stride=1)
    second = x.kernel_map(kernel_size=3, stride=1)

    assert first is second

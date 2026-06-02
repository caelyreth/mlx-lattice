from __future__ import annotations

import mlx_lattice


def test_native_backend_info() -> None:
    info = mlx_lattice.backend_info()

    assert info['version'] == mlx_lattice.__version__
    assert info['capabilities'] == {
        'cpu': True,
        'metal': False,
        'cuda': False,
        'rocm': False,
    }


def test_native_add_ints() -> None:
    assert mlx_lattice.add_ints(20, 22) == 42

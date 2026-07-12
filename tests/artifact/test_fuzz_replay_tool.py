from __future__ import annotations

import tarfile
from pathlib import Path

import mlx.core as mx
from lattice_conformance.archive import fixture_root
from lattice_conformance.metrics import distribution
from lattice_conformance.replay import _assert_sparse_close

from mlx_lattice import SparseTensor


def test_fuzz_replay_distribution_reports_common_accuracy_quantiles() -> (
    None
):
    stats = distribution([0.0, 1.0, 2.0, 3.0, 4.0])

    assert stats == {
        'avg': 2.0,
        'median': 2.0,
        'p95': 3.8,
        'p99': 3.96,
        'max': 4.0,
    }


def test_fuzz_replay_distribution_handles_empty_input() -> None:
    assert distribution([]) == {
        'avg': 0.0,
        'median': 0.0,
        'p95': 0.0,
        'p99': 0.0,
        'max': 0.0,
    }


def test_fixture_root_cleans_archive_extraction(tmp_path: Path) -> None:
    source = tmp_path / 'fixtures'
    source.mkdir()
    (source / 'manifest.json').write_text('{}\n', encoding='utf-8')
    archive = tmp_path / 'fixtures.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        handle.add(source, arcname=source.name)

    with fixture_root(archive) as extracted:
        temporary_root = extracted.parent
        assert (extracted / 'manifest.json').exists()

    assert not temporary_root.exists()


def test_sparse_replay_aligns_features_by_coordinate_value() -> None:
    expected_coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=mx.int32
    )
    expected_features = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    actual = SparseTensor(
        mx.array(
            [[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[3.0], [1.0], [2.0]], dtype=mx.float32),
    )

    max_abs, max_rel = _assert_sparse_close(
        actual,
        {
            'output.active': mx.array([3], dtype=mx.int32),
            'output.coords': expected_coords,
            'output.features': expected_features,
        },
        rtol=0.0,
        atol=0.0,
    )

    assert max_abs == 0.0
    assert max_rel == 0.0

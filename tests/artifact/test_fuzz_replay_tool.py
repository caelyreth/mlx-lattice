from __future__ import annotations

import tarfile
from pathlib import Path

from lattice_conformance.archive import fixture_root
from lattice_conformance.metrics import distribution


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

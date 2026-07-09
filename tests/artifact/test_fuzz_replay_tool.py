from __future__ import annotations

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

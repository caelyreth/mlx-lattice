from __future__ import annotations

import statistics


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            'avg': 0.0,
            'median': 0.0,
            'p95': 0.0,
            'p99': 0.0,
            'max': 0.0,
        }
    ordered = sorted(values)
    return {
        'avg': statistics.fmean(ordered),
        'median': statistics.median(ordered),
        'p95': percentile(ordered, 0.95),
        'p99': percentile(ordered, 0.99),
        'max': ordered[-1],
    }


def percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = ['distribution', 'percentile']

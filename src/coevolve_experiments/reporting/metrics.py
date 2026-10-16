"""Small, dependency-free statistics used by the paper tables."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def mean(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return statistics.fmean(selected) if selected else None


def sample_std(values: Iterable[float | int | None]) -> float | None:
    """Sample standard deviation, with a single observation defined as zero."""

    selected = [float(value) for value in values if value is not None]
    if not selected:
        return None
    return statistics.stdev(selected) if len(selected) > 1 else 0.0


def median_iqr(values: Iterable[float | int | None]) -> dict[str, float | None]:
    selected = sorted(float(value) for value in values if value is not None)
    if not selected:
        return {"median": None, "q1": None, "q3": None, "iqr": None}
    median = statistics.median(selected)
    if len(selected) == 1:
        q1 = q3 = selected[0]
    else:
        midpoint = len(selected) // 2
        lower = selected[:midpoint]
        upper = selected[-midpoint:]
        q1 = statistics.median(lower)
        q3 = statistics.median(upper)
    return {"median": median, "q1": q1, "q3": q3, "iqr": q3 - q1}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)

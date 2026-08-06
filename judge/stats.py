"""Spread and interval estimation for judge metrics.

A single-run kappa on 200 pairs is not a safe basis for ranking: at the flip
rates measured for this model class, run-to-run judge noise (~±0.05) is nearly
as large as the finite-sample uncertainty of having only 200 pairs (~±0.06).
Two models whose true kappas differ by less than roughly 0.15 will swap places
across single runs, so every metric here is reported with a spread and the
leaderboard refuses to rank overlapping intervals.
"""

from __future__ import annotations

import random
import statistics
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

BOOTSTRAP_RESAMPLES = 1000  # the standard for this kind of interval
CI_ALPHA = 0.05  # 95% interval


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation. sd is 0.0 for a single value.

    Callers must report n alongside: sd=0.0 from one run means "unmeasured",
    not "no variance".
    """
    if not values:
        raise ValueError("mean_sd() needs at least one value")
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.stdev(values)


def bootstrap_ci(
    items: Sequence[T],
    *,
    statistic: Callable[[list[T]], float],
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = CI_ALPHA,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling whole items with replacement.

    The resampling unit is the item, matching how the metrics aggregate. `seed`
    is required rather than defaulted: an unseeded interval would move every
    time the same results were scored.
    """
    if not items:
        raise ValueError("bootstrap_ci() needs at least one item")
    rng = random.Random(seed)
    n = len(items)
    estimates = sorted(
        statistic([items[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    lo = estimates[int((alpha / 2) * resamples)]
    hi = estimates[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return lo, hi


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if two intervals share any point — touching endpoints included.

    Touching is not separation, so a touching pair stays unranked.
    """
    return a[0] <= b[1] and b[0] <= a[1]

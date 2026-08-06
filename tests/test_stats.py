import pytest

from judge.stats import bootstrap_ci, intervals_overlap, mean_sd


def test_mean_sd_of_repeated_runs():
    mean, sd = mean_sd([0.60, 0.70, 0.65])
    assert mean == pytest.approx(0.65)
    assert sd == pytest.approx(0.05)


def test_mean_sd_of_single_run_has_no_spread():
    # One run tells you nothing about run-to-run variance. Reporting sd=0 would
    # claim it does, so callers must see it as "unmeasured" via n.
    assert mean_sd([0.42]) == (pytest.approx(0.42), 0.0)


def test_mean_sd_rejects_empty():
    with pytest.raises(ValueError):
        mean_sd([])


def test_bootstrap_ci_brackets_the_point_estimate():
    items = [1.0] * 60 + [0.0] * 40
    lo, hi = bootstrap_ci(items, statistic=lambda xs: sum(xs) / len(xs), seed=7)
    assert lo < 0.60 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_is_deterministic_for_a_given_seed():
    items = [1.0] * 30 + [0.0] * 20
    stat = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert bootstrap_ci(items, statistic=stat, seed=99) == bootstrap_ci(items, statistic=stat, seed=99)


def test_bootstrap_ci_differs_across_seeds():
    # Continuous values, not 0/1: with binary data the percentile endpoints
    # land on the same discrete grid for most seeds and the test would pass
    # even if the seed were ignored entirely.
    items = [i / 50 for i in range(50)]
    stat = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert bootstrap_ci(items, statistic=stat, seed=1) != bootstrap_ci(items, statistic=stat, seed=2)


def test_bootstrap_ci_narrows_as_n_grows():
    stat = lambda xs: sum(xs) / len(xs)  # noqa: E731
    small_lo, small_hi = bootstrap_ci([1.0, 0.0] * 15, statistic=stat, seed=3)
    large_lo, large_hi = bootstrap_ci([1.0, 0.0] * 300, statistic=stat, seed=3)
    assert (large_hi - large_lo) < (small_hi - small_lo)


def test_bootstrap_ci_resamples_whole_items():
    # Item-level cluster resampling: the unit is the item, so a statistic that
    # counts items must always see exactly len(items) of them.
    seen = []
    bootstrap_ci(
        [("p1", 1), ("p2", 0), ("p3", 1)],
        statistic=lambda xs: (seen.append(len(xs)), 0.0)[1],
        seed=5,
        resamples=10,
    )
    assert seen == [3] * 10


def test_intervals_overlap():
    assert intervals_overlap((0.40, 0.60), (0.55, 0.70))
    assert intervals_overlap((0.40, 0.60), (0.10, 0.45))
    assert not intervals_overlap((0.40, 0.60), (0.61, 0.80))
    assert not intervals_overlap((0.61, 0.80), (0.40, 0.60))


def test_intervals_that_merely_touch_count_as_overlapping():
    # Touching endpoints are not a separation; ranking on them would claim a
    # distinction the data does not support.
    assert intervals_overlap((0.40, 0.60), (0.60, 0.80))

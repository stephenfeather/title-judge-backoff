"""Issue #43: reliability is a selection criterion, so it must be a reported metric."""

from judge.reliability import (
    coverage_shapes,
    reliability_rows,
    render_coverage_bias_caveat,
    render_reliability_section,
)
from judge.schema import ReasonCode, Verdict


def verdict(pair_id, model="m"):
    return Verdict(
        pair_id=pair_id,
        verdict="approve",
        reason=ReasonCode.OK,
        model_id=model,
        prompt_version="v2",
        temperature=0.0,
        run_index=0,
    )


def rows(pair_ids):
    return [verdict(p) for p in pair_ids]


def manifest(**health):
    return {"health": {"calls_ok": 0, "calls_failed": 0, "error_kinds": {}, **health}}


# --- what the evidence covers -------------------------------------------------


def test_a_cumulative_block_is_preferred_over_the_last_launch():
    # #40 gave manifests a cumulative block. Reading `health` when cumulative
    # exists reports the smallest, final resume segment as the whole run.
    m = {
        "health": {"calls_ok": 24, "calls_failed": 0, "error_kinds": {}},
        "cumulative": {
            "launches": 3,
            "calls_ok": 600,
            "calls_failed": 91,
            "error_kinds": {"ReadTimeout": 91},
        },
    }
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": m})
    assert row.scope == "cumulative"
    assert (row.attempted, row.succeeded, row.failed) == (691, 600, 91)
    assert row.launches == 3


def test_a_manifest_without_cumulative_is_labelled_last_launch_only():
    # The 2026-08-11 run: every manifest predates #40. nemotron's says 24 calls,
    # 0 failed, for a backend with ~600 rows on disk. Reporting that as the run's
    # reliability would be a false clean bill of health.
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": manifest(calls_ok=24)})
    assert row.scope == "last-launch"


def test_a_backend_with_no_manifest_reports_unknown_not_zero():
    # kimi-k2.6 in the v2 run: 255 rows on disk, killed before it could write a
    # manifest. Zero recorded failures is the ABSENCE of evidence about the
    # backend whose failures mattered most — it must never render as a clean run.
    (row,) = reliability_rows({"kimi": rows(["p1", "p2"])}, {})
    assert row.scope == "none"
    assert row.attempted is None
    assert row.failed is None
    assert row.failure_rate is None
    assert row.rows_on_disk == 2


# --- how it failed ------------------------------------------------------------


def test_timeouts_are_counted_apart_from_other_failures():
    # "attempted N, failed M, of which K were timeouts" is the issue's wording.
    # A timeout is the failure that decides usability at pack scale; lumping it
    # with a 500 loses the distinction that matters.
    m = {"cumulative": {"calls_ok": 10, "calls_failed": 7, "error_kinds": {
        "ReadTimeout": 4, "ConnectTimeout": 1, "HTTPStatusError": 2}}}
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": m})
    assert row.timeouts == 5
    assert row.contract_failures == 0


def test_contract_failures_are_counted_apart_from_transport_failures():
    # #41: answering in the wrong shape is evidence about the MODEL. A dropped
    # connection is not. They must not share a column.
    m = {"cumulative": {"calls_ok": 10, "calls_failed": 3, "error_kinds": {
        "JudgeResponseError": 3}}}
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": m})
    assert row.contract_failures == 3
    assert row.timeouts == 0


def test_failure_rate_and_attempts_per_vote_are_derived_from_the_counts():
    m = {"cumulative": {"calls_ok": 75, "calls_failed": 25, "error_kinds": {"ReadTimeout": 25}}}
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": m})
    assert row.failure_rate == 0.25
    assert row.attempts_per_vote == 100 / 75


def test_a_backend_that_fails_more_than_half_its_calls_is_marked_unusable():
    # The issue's second acceptance criterion. Not a tuned threshold: at >=2
    # attempts per completed vote a backend needs more calls than the set has
    # votes, so the set costs more than double and grows with every resume.
    m = {"cumulative": {"calls_ok": 100, "calls_failed": 100, "error_kinds": {"ReadTimeout": 100}}}
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": m})
    assert row.unusable_at_scale is True


def test_a_healthy_backend_is_not_marked_unusable():
    m = {"cumulative": {"calls_ok": 600, "calls_failed": 3, "error_kinds": {"ReadTimeout": 3}}}
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": m})
    assert row.unusable_at_scale is False


def test_unusable_is_unknown_rather_than_false_without_evidence():
    # Absent is not "fine". A backend with no manifest cannot be cleared.
    (row,) = reliability_rows({"a": rows(["p1"])}, {})
    assert row.unusable_at_scale is None


# --- latency ------------------------------------------------------------------


def test_latency_percentiles_come_from_the_latest_launch_and_say_so():
    # p50/p95 cannot be summed across launches — a median is not mergeable
    # (#40). The latest launch's distribution is reported as its own thing.
    m = {
        "health": {"calls_ok": 5, "calls_failed": 0, "error_kinds": {},
                   "latency_median": 12.0, "latency_p95": 63.2},
        "cumulative": {"calls_ok": 600, "calls_failed": 0, "error_kinds": {}},
    }
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": m})
    assert (row.latency_p50, row.latency_p95) == (12.0, 63.2)


def test_a_manifest_written_before_p95_existed_reports_it_absent():
    (row,) = reliability_rows({"a": rows(["p1"])}, {"a": manifest(latency_median=1.0)})
    assert row.latency_p50 == 1.0
    assert row.latency_p95 is None


# --- rendering ----------------------------------------------------------------


def test_the_section_names_the_unusable_backend_in_prose():
    m = {"cumulative": {"calls_ok": 100, "calls_failed": 100, "error_kinds": {"ReadTimeout": 100}}}
    text = "\n".join(render_reliability_section(reliability_rows({"slow": rows(["p1"])}, {"slow": m})))
    assert "slow" in text
    assert "unusable" in text.lower()


def test_the_section_says_which_backends_have_no_evidence():
    text = "\n".join(render_reliability_section(reliability_rows({"kimi": rows(["p1"])}, {})))
    assert "kimi" in text
    assert "unknown" in text.lower()


def test_the_section_is_empty_without_backends():
    assert render_reliability_section([]) == []


# --- coverage shape: prefix, not sample ---------------------------------------


def test_a_partial_backend_that_occupies_a_contiguous_prefix_is_identified():
    # Votes are attempted in pairs-file order, so a backend that degrades
    # partway through completes a PREFIX. The calibration file is cohort-ordered
    # (most-changed, then quarantined-brand, then representative, then
    # unchanged), so a prefix is a cohort skew, not a sample.
    order = [f"p{i}" for i in range(10)]
    shapes = coverage_shapes({"full": rows(order), "partial": rows(order[:3])})
    partial = next(s for s in shapes if s.backend == "partial")
    assert partial.complete is False
    assert partial.span == 3
    assert partial.is_prefix is True


def test_a_partial_backend_spread_across_the_set_is_not_called_a_prefix():
    # An honest distinction: rows scattered over the whole file ARE closer to a
    # sample, and saying "prefix" there would be a claim the data refutes.
    order = [f"p{i}" for i in range(10)]
    shapes = coverage_shapes({"full": rows(order), "spread": rows(["p0", "p4", "p9"])})
    spread = next(s for s in shapes if s.backend == "spread")
    assert spread.span == 10
    assert spread.is_prefix is False


def test_a_complete_backend_carries_no_bias_caveat():
    order = [f"p{i}" for i in range(5)]
    shapes = coverage_shapes({"a": rows(order), "b": rows(order)})
    assert all(s.complete for s in shapes)
    assert render_coverage_bias_caveat(shapes) == []


def test_the_bias_caveat_names_the_partial_backend_and_its_counts():
    order = [f"p{i}" for i in range(10)]
    shapes = coverage_shapes({"full": rows(order), "partial": rows(order[:3])})
    lines = render_coverage_bias_caveat(shapes)
    text = "\n".join(lines)
    assert "partial" in text
    assert "3" in text and "10" in text
    # The complete backend is not accused: it appears in no table row.
    named = [line for line in lines if line.startswith("| ") and "Backend" not in line]
    assert not any("full" in line for line in named)


def test_reference_order_is_the_most_complete_backend():
    # Whichever backend saw the most pairs defines the file order. Anything less
    # would rank against a partial view and mislabel a prefix.
    shapes = coverage_shapes({
        "short": rows(["p0", "p1"]),
        "long": rows(["p0", "p1", "p2", "p3"]),
    })
    assert {s.total_pairs for s in shapes} == {4}


def test_coverage_shapes_is_empty_without_verdicts():
    assert coverage_shapes({}) == []

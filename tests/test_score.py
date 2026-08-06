import pytest

from judge.schema import Pair, ReasonCode, Verdict
from score import cohens_kappa, render_leaderboard, score_model


def make_pair(pair_id, ground_truth, reason):
    return Pair(
        id=pair_id,
        original=f"orig {pair_id}",
        enriched=f"enriched {pair_id}",
        brand="Acme",
        mpn=f"MPN-{pair_id}",
        ground_truth=ground_truth,
        reason=ReasonCode(reason),
    )


def make_verdict(pair_id, verdict, reason, model_id="model-a", run_index=0):
    return Verdict(
        pair_id=pair_id,
        verdict=verdict,
        reason=ReasonCode(reason),
        model_id=model_id,
        prompt_version="v1",
        temperature=None,
        run_index=run_index,
    )


PAIRS = [
    make_pair("p1", "approve", "ok"),
    make_pair("p2", "approve", "ok"),
    make_pair("p3", "reject", "meaning_change"),
    make_pair("p4", "reject", "casing_error"),
]


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa(["approve", "reject"], ["approve", "reject"]) == 1.0


def test_cohens_kappa_chance_agreement_is_zero():
    truth = ["approve", "approve", "reject", "reject"]
    pred = ["approve", "reject", "approve", "reject"]
    assert cohens_kappa(truth, pred) == 0.0


def test_score_model_metrics():
    verdicts = [
        make_verdict("p1", "approve", "ok"),
        make_verdict("p2", "reject", "overcorrection"),
        make_verdict("p3", "reject", "meaning_change"),
        make_verdict("p4", "approve", "ok"),
    ]
    result = score_model(PAIRS, verdicts)
    assert result.model_id == "model-a"
    assert result.n == 4
    assert result.accuracy == 0.5
    assert result.false_approve_rate == 0.5  # p4: truth reject, judged approve
    assert result.reason_confusion[("ok", "overcorrection")] == 1
    assert result.reason_confusion[("meaning_change", "meaning_change")] == 1


def test_score_model_ignores_verdicts_for_unknown_pairs():
    verdicts = [
        make_verdict("p1", "approve", "ok"),
        make_verdict("p999", "approve", "ok"),
    ]
    result = score_model(PAIRS, verdicts)
    assert result.n == 1


def test_score_model_reports_coverage():
    verdicts = [make_verdict("p1", "approve", "ok")]
    result = score_model(PAIRS, verdicts)
    assert result.coverage == 0.25


def test_partial_coverage_backend_is_excluded_from_ranking():
    # Judged only the one pair it gets right: perfect kappa on 25% coverage.
    partial = [make_verdict("p1", "approve", "ok", model_id="model-partial")]
    # Judged everything, one mistake.
    full = [
        make_verdict("p1", "approve", "ok", model_id="model-full"),
        make_verdict("p2", "approve", "ok", model_id="model-full"),
        make_verdict("p3", "reject", "meaning_change", model_id="model-full"),
        make_verdict("p4", "approve", "ok", model_id="model-full"),
    ]
    md = render_leaderboard([score_model(PAIRS, partial), score_model(PAIRS, full)])
    ranked_table = md.split("## ")[0]
    assert "model-full" in ranked_table
    assert "model-partial" not in ranked_table
    assert "model-partial" in md  # still visible, marked as excluded
    assert "coverage" in md.lower()


def test_render_leaderboard_sorted_by_kappa():
    perfect = [make_verdict(p.id, p.ground_truth, p.reason.value, model_id="model-good") for p in PAIRS]
    inverted = [
        make_verdict(p.id, "reject" if p.ground_truth == "approve" else "approve", "ok", model_id="model-bad")
        for p in PAIRS
    ]
    scores = [score_model(PAIRS, inverted), score_model(PAIRS, perfect)]
    md = render_leaderboard(scores)
    assert md.index("model-good") < md.index("model-bad")
    assert "kappa" in md.lower()


# --- majority voting, spread, and flip rates -------------------------------


def votes_for(pair_id, rulings, model_id="model-a"):
    """One verdict per (verdict, reason) ruling, numbered as successive votes."""
    return [
        make_verdict(pair_id, verdict, reason, model_id=model_id, run_index=i)
        for i, (verdict, reason) in enumerate(rulings)
    ]


def test_score_model_scores_the_majority_verdict_not_each_vote():
    # p1 is judged reject twice and approve once: the run counts as reject, and
    # n counts pairs, not calls.
    verdicts = (
        votes_for("p1", [("reject", "ok"), ("approve", "ok"), ("reject", "ok")])
        + votes_for("p2", [("approve", "ok")] * 3)
        + votes_for("p3", [("reject", "meaning_change")] * 3)
        + votes_for("p4", [("reject", "casing_error")] * 3)
    )
    result = score_model(PAIRS, verdicts)
    assert result.n == 4
    assert result.n_votes == 3
    assert result.accuracy == 0.75  # only p1 wrong (truth approve, majority reject)


def test_score_model_reports_flip_rates_and_names_the_unstable_items():
    # R5: flip rates are a health metric, and the flipping items are the
    # borderline set worth sharpening the rubric on.
    verdicts = (
        votes_for("p1", [("approve", "ok"), ("approve", "ok"), ("reject", "ok")])
        + votes_for("p2", [("approve", "ok")] * 3)
        + votes_for("p3", [("reject", "meaning_change")] * 3)
        + votes_for("p4", [("reject", "casing_error")] * 3)
    )
    result = score_model(PAIRS, verdicts)
    assert result.verdict_flip_rate == pytest.approx((1 / 3) / 4)
    assert result.unstable_pair_ids == ["p1"]


def test_score_model_tracks_reason_flips_even_when_the_verdict_is_stable():
    # The probe found exactly this shape: reject 9/9, but the reason code moved.
    # A single flip number would have reported this run as perfectly stable.
    verdicts = (
        votes_for("p3", [("reject", "meaning_change"), ("reject", "casing_error"), ("reject", "meaning_change")])
        + votes_for("p1", [("approve", "ok")] * 3)
        + votes_for("p2", [("approve", "ok")] * 3)
        + votes_for("p4", [("reject", "casing_error")] * 3)
    )
    result = score_model(PAIRS, verdicts)
    assert result.verdict_flip_rate == 0.0
    assert result.reason_flip_rate == pytest.approx((1 / 3) / 4)
    assert result.unstable_pair_ids == ["p3"]


def test_score_model_reports_kappa_spread_across_runs():
    # Run 0 gets p1 wrong, runs 1 and 2 are perfect: three different per-run
    # kappas, so sd must be non-zero and the mean must sit between them.
    verdicts = (
        votes_for("p1", [("reject", "ok"), ("approve", "ok"), ("approve", "ok")])
        + votes_for("p2", [("approve", "ok")] * 3)
        + votes_for("p3", [("reject", "meaning_change")] * 3)
        + votes_for("p4", [("reject", "casing_error")] * 3)
    )
    result = score_model(PAIRS, verdicts)
    assert result.n_votes == 3
    assert result.kappa_sd > 0.0
    assert result.kappa_run_mean < 1.0


def test_single_vote_run_reports_unmeasured_spread():
    # With N=1 there is no run-to-run evidence. sd=0.0 here means "unmeasured",
    # which is why n_votes is reported next to it.
    verdicts = [make_verdict(p.id, p.ground_truth, p.reason.value) for p in PAIRS]
    result = score_model(PAIRS, verdicts)
    assert result.n_votes == 1
    assert result.kappa_sd == 0.0
    assert result.verdict_flip_rate == 0.0


def test_score_model_reports_a_bootstrap_interval_around_kappa():
    verdicts = [make_verdict(p.id, p.ground_truth, p.reason.value) for p in PAIRS]
    result = score_model(PAIRS, verdicts)
    lo, hi = result.kappa_ci
    assert lo <= result.kappa <= hi


def test_leaderboard_refuses_to_rank_models_with_overlapping_intervals():
    # R4: two models whose CIs overlap are not separable at this sample size.
    # Printing them 1st and 2nd would present noise as a finding.
    good = [make_verdict(p.id, p.ground_truth, p.reason.value, model_id="model-good") for p in PAIRS]
    nearly = [
        make_verdict("p1", "approve", "ok", model_id="model-nearly"),
        make_verdict("p2", "reject", "ok", model_id="model-nearly"),
        make_verdict("p3", "reject", "meaning_change", model_id="model-nearly"),
        make_verdict("p4", "reject", "casing_error", model_id="model-nearly"),
    ]
    md = render_leaderboard([score_model(PAIRS, good), score_model(PAIRS, nearly)])
    assert "not separable" in md.lower()


def test_score_model_refuses_to_score_unruled_pairs():
    # The whole point of the guard: None is not a label. Scoring it would
    # produce a kappa built on "None vs approve" comparisons and report a
    # confident number derived from data nobody has ruled.
    unruled = [
        Pair(id="u1", original="a", enriched="b"),
        Pair(id="u2", original="c", enriched="d"),
    ]
    verdicts = [make_verdict("u1", "approve", "ok"), make_verdict("u2", "reject", "ok")]
    with pytest.raises(ValueError, match="unruled"):
        score_model(unruled, verdicts)


def test_score_model_refuses_a_partially_ruled_set():
    # Scoring only the ruled subset would silently change the denominator and
    # report coverage against a set that is not the one requested.
    mixed = [make_pair("p1", "approve", "ok"), Pair(id="u1", original="a", enriched="b")]
    verdicts = [make_verdict("p1", "approve", "ok"), make_verdict("u1", "approve", "ok")]
    with pytest.raises(ValueError, match="unruled"):
        score_model(mixed, verdicts)


def test_leaderboard_shows_spread_and_flip_columns():
    verdicts = (
        votes_for("p1", [("approve", "ok"), ("approve", "ok"), ("reject", "ok")])
        + votes_for("p2", [("approve", "ok")] * 3)
        + votes_for("p3", [("reject", "meaning_change")] * 3)
        + votes_for("p4", [("reject", "casing_error")] * 3)
    )
    md = render_leaderboard([score_model(PAIRS, verdicts)])
    assert "sd" in md.lower()
    assert "95% ci" in md.lower()
    assert "flip" in md.lower()
    assert "votes" in md.lower()

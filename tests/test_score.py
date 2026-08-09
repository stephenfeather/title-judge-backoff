import random

from dataclasses import replace

import pytest

from judge.schema import Pair, ReasonCode, Verdict, verdict_to_json_line
from judge.stats import intervals_overlap
from judge.vote import tally_votes
from score import (
    ModelScore,
    cohens_kappa,
    render_leaderboard,
    score_model,
    separability_tiers,
)


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


def test_score_model_refuses_a_file_holding_two_models():
    # Issue #16: model_id was taken from known[0], so a results file produced by
    # concatenating two backends' shards — a merge tool, a `cat` of two files, a
    # recovery script — scored fine and was labelled with whichever row sorted
    # first. already_judged_ids guards the WRITER; nothing guarded the reader.
    verdicts = [
        make_verdict("p1", "approve", "ok", model_id="model-a"),
        make_verdict("p2", "reject", "overcorrection", model_id="model-b"),
    ]
    with pytest.raises(ValueError) as exc:
        score_model(PAIRS, verdicts)
    # Naming both is the point: "two models" alone does not say which file to fix.
    assert "model-a" in str(exc.value)
    assert "model-b" in str(exc.value)


def test_score_model_names_every_model_it_found_not_just_two():
    verdicts = [
        make_verdict("p1", "approve", "ok", model_id="model-a"),
        make_verdict("p2", "reject", "overcorrection", model_id="model-b"),
        make_verdict("p3", "reject", "meaning_change", model_id="model-c"),
    ]
    with pytest.raises(ValueError) as exc:
        score_model(PAIRS, verdicts)
    assert all(m in str(exc.value) for m in ("model-a", "model-b", "model-c"))


def test_score_model_refuses_a_file_mixing_prompt_versions():
    # The same defect for a sibling field, and live as of #14: results/ holds v1
    # files and new runs write v2, so a stitched-together file mixing them is a
    # real possibility now. Averaging two prompts is averaging two instruments.
    verdicts = [
        replace(make_verdict("p1", "approve", "ok"), prompt_version="v1"),
        replace(make_verdict("p2", "reject", "overcorrection"), prompt_version="v2"),
    ]
    with pytest.raises(ValueError) as exc:
        score_model(PAIRS, verdicts)
    assert "prompt_version" in str(exc.value)
    assert "v1" in str(exc.value) and "v2" in str(exc.value)


def test_score_model_refuses_a_file_mixing_reasoning_effort():
    # Effort changes the reason code the model returns, so two efforts in one
    # file are two different judges sharing a filename.
    verdicts = [
        replace(make_verdict("p1", "approve", "ok"), reasoning_effort="none"),
        replace(make_verdict("p2", "reject", "overcorrection"), reasoning_effort="medium"),
    ]
    with pytest.raises(ValueError) as exc:
        score_model(PAIRS, verdicts)
    assert "reasoning_effort" in str(exc.value)


def test_score_model_is_unchanged_for_a_normal_single_model_file():
    verdicts = [
        make_verdict("p1", "approve", "ok"),
        make_verdict("p2", "reject", "overcorrection"),
    ]
    assert score_model(PAIRS, verdicts).model_id == "model-a"


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


def make_score(model_id, kappa, ci):
    """A ModelScore with only the fields separability_tiers reads."""
    return ModelScore(
        model_id=model_id,
        n=4,
        coverage=1.0,
        n_unsettled_verdict=0,
        n_unsettled_reason=0,
        accuracy=1.0,
        kappa=kappa,
        false_approve_rate=0.0,
        reason_confusion={},
        n_votes=1,
        kappa_run_mean=kappa,
        kappa_sd=0.0,
        kappa_ci=ci,
        verdict_flip_rate=0.0,
        reason_flip_rate=0.0,
        unstable_pair_ids=[],
    )


def test_tiers_never_separate_two_models_whose_intervals_overlap():
    # THE invariant. Walking down by kappa and comparing only against the most
    # recent tier misses chained overlap: A does not overlap B, so B opens a new
    # tier; C overlaps B and joins it — leaving A and C in different tiers and
    # implying A > C, even though A and C DO overlap and are not separable.
    a = make_score("A", 0.85, (0.80, 0.90))
    b = make_score("B", 0.65, (0.60, 0.70))
    c = make_score("C", 0.60, (0.50, 0.85))  # overlaps A and B, but not vice versa
    assert intervals_overlap(a.kappa_ci, c.kappa_ci)
    assert not intervals_overlap(a.kappa_ci, b.kappa_ci)

    tiers = separability_tiers([a, b, c])
    tier_of = {s.model_id: i for i, tier in enumerate(tiers) for s in tier}
    assert tier_of["A"] == tier_of["C"], "A and C overlap — they cannot be in different tiers"

    # Stated as the general invariant: any two models in DIFFERENT tiers must
    # have non-overlapping intervals.
    for i, left_tier in enumerate(tiers):
        for j, right_tier in enumerate(tiers):
            if i == j:
                continue
            for left in left_tier:
                for right in right_tier:
                    assert not intervals_overlap(left.kappa_ci, right.kappa_ci), (
                        f"{left.model_id} and {right.model_id} overlap but are in different tiers"
                    )


def test_tiers_still_separate_genuinely_disjoint_models():
    # The fix must not collapse everything into one tier.
    high = make_score("high", 0.90, (0.85, 0.95))
    low = make_score("low", 0.20, (0.10, 0.30))
    tiers = separability_tiers([high, low])
    assert [[s.model_id for s in t] for t in tiers] == [["high"], ["low"]]


def test_tiers_are_ordered_best_first():
    a = make_score("a", 0.90, (0.85, 0.95))
    b = make_score("b", 0.50, (0.45, 0.55))
    c = make_score("c", 0.20, (0.10, 0.30))
    tiers = separability_tiers([c, a, b])
    assert [s.model_id for t in tiers for s in t] == ["a", "b", "c"]


def test_n_votes_reflects_the_runs_the_spread_was_computed_over():
    # n_votes sits beside kappa_sd, and sd is the spread of PER-RUN kappas — so
    # n_votes must be the number of runs, not the max votes any single pair got.
    # p1 has 2 votes and p2 has 1, but three distinct run indices exist.
    verdicts = [
        make_verdict("p1", "approve", "ok", run_index=0),
        make_verdict("p1", "approve", "ok", run_index=1),
        make_verdict("p2", "approve", "ok", run_index=2),
        make_verdict("p3", "reject", "meaning_change", run_index=0),
        make_verdict("p4", "reject", "casing_error", run_index=0),
    ]
    result = score_model(PAIRS, verdicts)
    assert result.n_votes == 3


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


# --------------------------------------------------------------------------
# Line-order independence (issue #9 acceptance 5)
#
# A serial run always appends verdicts in pair-then-vote order, and scoring
# quietly leaned on that. Concurrent workers make the order arbitrary, so the
# same verdicts must score identically however they land in the file.
# --------------------------------------------------------------------------


ORDER_VERDICTS = (
    votes_for("p1", [("approve", "ok"), ("reject", "meaning_change"), ("approve", "ok")])
    + votes_for("p2", [("approve", "ok"), ("approve", "ok"), ("reject", "casing_error")])
    # p3 uses the shape a real sweep produces most often:
    # verdict settled 3-0, reason split three ways. Only the tie-break decides
    # its reason, so any order dependence shows up here first. `reason` has five
    # codes and demonstrably splits three ways in production; `verdict` is
    # binary and cannot, which is why the reason column is the real exposure.
    + votes_for(
        "p3",
        [("reject", "meaning_change"), ("reject", "ok"), ("reject", "overcorrection")],
    )
    + votes_for("p4", [("reject", "casing_error"), ("approve", "ok"), ("reject", "casing_error")])
)


def test_score_model_is_independent_of_verdict_order():
    baseline = score_model(PAIRS, list(ORDER_VERDICTS))
    for seed in range(8):
        shuffled = list(ORDER_VERDICTS)
        random.Random(seed).shuffle(shuffled)
        assert score_model(PAIRS, shuffled) == baseline, f"seed {seed} scored differently"


def test_scoring_a_shuffled_results_file_is_byte_identical(tmp_path):
    # Acceptance 5, end to end and through the file layer: write a results
    # file, shuffle its LINES, score both, compare every reported number.
    #
    # This is not hypothetical. Resume already breaks pair-then-vote order
    # today with no concurrency involved: when a vote fails, its retry appends
    # at the end of the file on the next launch. In the 2026-08-06 S1 run that
    # left 47 of deepseek's 200 pairs with their votes out of run_index order,
    # and it silently decided the reason code on two three-way ties.
    lines = [verdict_to_json_line(v) for v in ORDER_VERDICTS]

    ordered_path = tmp_path / "ordered.jsonl"
    shuffled_path = tmp_path / "shuffled.jsonl"
    ordered_path.write_text("".join(f"{line}\n" for line in lines))

    scrambled = list(lines)
    random.Random(20260806).shuffle(scrambled)
    assert scrambled != lines, "fixture did not actually shuffle"
    shuffled_path.write_text("".join(f"{line}\n" for line in scrambled))

    from score import load_verdicts, render_leaderboard

    ordered_score = score_model(PAIRS, load_verdicts(ordered_path))
    shuffled_score = score_model(PAIRS, load_verdicts(shuffled_path))

    assert shuffled_score == ordered_score
    # Including the CI, which a seeded bootstrap over a reordered sequence
    # would silently move, and the rendered report the operator actually reads.
    assert shuffled_score.kappa_ci == ordered_score.kappa_ci
    assert render_leaderboard([shuffled_score]) == render_leaderboard([ordered_score])


def test_resume_style_reordering_does_not_change_the_ruling():
    # The exact shape resume produces: a failed vote retried on a later launch
    # lands AFTER every other pair's votes rather than beside its siblings.
    # The shape resume produces: a retried vote 0 lands at file
    # position 521 while its run_index 1 and 2 sat at positions 28 and 29.
    in_order = list(ORDER_VERDICTS)
    late, rest = [], []
    for v in in_order:
        (late if v.run_index == 0 else rest).append(v)
    resumed = rest + late  # every vote 0 retried at the end

    assert score_model(PAIRS, resumed) == score_model(PAIRS, in_order)


def test_unsettled_verdict_is_excluded_from_kappa_and_counted():
    # A 1-1 verdict split has no majority. Scoring it would be scoring the
    # tie-break, and false_approve_rate is a SAFETY number — a coin flip
    # recorded as an approve directly corrupts the metric that governs
    # deployment risk. Excluded, and the exclusion is reported.
    verdicts = (
        votes_for("p1", [("approve", "ok"), ("reject", "meaning_change")])  # tied 1-1
        + votes_for("p2", [("approve", "ok")] * 3)
        + votes_for("p3", [("reject", "meaning_change")] * 3)
        + votes_for("p4", [("reject", "casing_error")] * 3)
    )
    result = score_model(PAIRS, verdicts)
    assert result.n == 3, "the tied pair must not be scored"
    assert result.n_unsettled_verdict == 1
    # Coverage still says the backend judged all four — that is a different
    # question from whether the votes decided, and the leaderboard's
    # partial-coverage gate must not be tripped by contested pairs.
    assert result.coverage == 1.0


def test_unsettled_reason_is_excluded_from_confusion_but_not_from_kappa():
    # The common shape: verdict settled 3-0, reason split 1-1-1. The verdict is
    # real evidence and must still be scored; only the reason is unusable.
    verdicts = (
        votes_for(
            "p3",
            [("reject", "meaning_change"), ("reject", "ok"), ("reject", "overcorrection")],
        )
        + votes_for("p1", [("approve", "ok")] * 3)
        + votes_for("p2", [("approve", "ok")] * 3)
        + votes_for("p4", [("reject", "casing_error")] * 3)
    )
    result = score_model(PAIRS, verdicts)
    assert result.n == 4, "a settled verdict is still evidence"
    assert result.n_unsettled_verdict == 0
    assert result.n_unsettled_reason == 1
    # p3's ground truth is meaning_change; no cell may be credited for it.
    assert not any(gt == "meaning_change" for gt, _ in result.reason_confusion)


def test_leaderboard_reports_the_excluded_pairs():
    # A metric over 3 of 4 pairs is honest; a silent exclusion is its own lie.
    verdicts = (
        votes_for("p1", [("approve", "ok"), ("reject", "meaning_change")])
        + votes_for("p2", [("approve", "ok")] * 3)
        + votes_for("p3", [("reject", "meaning_change")] * 3)
        + votes_for("p4", [("reject", "casing_error")] * 3)
    )
    md = render_leaderboard([score_model(PAIRS, verdicts)])
    assert "unsettled" in md.lower()


def test_a_model_whose_every_pair_is_unsettled_does_not_crash():
    # Degenerate but reachable on a backend that lost a vote on every pair.
    verdicts = [
        v
        for pid in ("p1", "p2", "p3", "p4")
        for v in votes_for(pid, [("approve", "ok"), ("reject", "casing_error")])
    ]
    with pytest.raises(ValueError, match="no pair"):
        score_model(PAIRS, verdicts)


def test_a_tied_verdict_is_reported_unsettled_rather_than_tie_broken():
    # PR #10 made this tie-break deterministic by run_index. Issue #12 goes
    # further: reproducible was never the same as correct, so there is now no
    # winner at all. Order-independence still holds — it just holds trivially.
    tied = [
        make_verdict("p1", "approve", "ok", run_index=0),
        make_verdict("p1", "reject", "meaning_change", run_index=1),
    ]
    forward = tally_votes(list(tied))
    backward = tally_votes(list(reversed(tied)))
    assert forward == backward
    assert forward[0].verdict is None
    assert forward[0].verdict_settled is False


def test_a_three_way_reason_split_is_reported_unsettled():
    tied = [
        make_verdict("p1", "reject", "meaning_change", run_index=0),
        make_verdict("p1", "reject", "casing_error", run_index=1),
        make_verdict("p1", "reject", "overcorrection", run_index=2),
    ]
    forward = tally_votes(list(tied))
    backward = tally_votes(list(reversed(tied)))
    assert forward == backward
    assert forward[0].reason is None
    # The verdict was unanimous and is still reported.
    assert forward[0].verdict == "reject"

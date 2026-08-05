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


def make_verdict(pair_id, verdict, reason, model_id="model-a"):
    return Verdict(
        pair_id=pair_id,
        verdict=verdict,
        reason=ReasonCode(reason),
        model_id=model_id,
        prompt_version="v1",
        temperature=0.0,
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

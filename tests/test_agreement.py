import pytest

from judge.agreement import (
    agreement_matrix,
    pairwise_agreement,
    reason_distribution,
    reason_cross_tab,
)
from judge.schema import ReasonCode, Verdict


def make_verdict(pair_id, verdict, reason, model_id="m", run_index=0):
    return Verdict(
        pair_id=pair_id,
        verdict=verdict,
        reason=ReasonCode(reason),
        model_id=model_id,
        prompt_version="v1",
        temperature=None,
        run_index=run_index,
    )


def model_votes(model_id, rulings):
    """One majority-ready vote per pair for a model: {pair_id: (verdict, reason)}."""
    return [
        make_verdict(pair_id, verdict, reason, model_id=model_id)
        for pair_id, (verdict, reason) in rulings.items()
    ]


A = model_votes("model-a", {"p1": ("approve", "ok"), "p2": ("reject", "casing_error"), "p3": ("reject", "meaning_change")})
B = model_votes("model-b", {"p1": ("approve", "ok"), "p2": ("reject", "truncation_worse"), "p3": ("approve", "ok")})


def test_pairwise_agreement_is_the_fraction_of_shared_pairs_that_match():
    # p1 and p2 agree on the verdict, p3 does not.
    assert pairwise_agreement(A, B) == pytest.approx(2 / 3)


def test_pairwise_agreement_is_symmetric():
    assert pairwise_agreement(A, B) == pairwise_agreement(B, A)


def test_pairwise_agreement_only_counts_pairs_both_models_judged():
    # A backend that errored on a pair must not drag another model's agreement
    # down — the comparison is over the intersection, and n says how big it was.
    partial = model_votes("model-c", {"p1": ("approve", "ok")})
    assert pairwise_agreement(A, partial) == 1.0


def test_pairwise_agreement_with_no_overlap_is_none():
    disjoint = model_votes("model-d", {"z9": ("approve", "ok")})
    assert pairwise_agreement(A, disjoint) is None


def test_agreement_matrix_covers_every_model_pair():
    matrix = agreement_matrix({"model-a": A, "model-b": B})
    assert matrix[("model-a", "model-b")] == pytest.approx(2 / 3)
    assert matrix[("model-a", "model-a")] == 1.0
    assert matrix[("model-b", "model-b")] == 1.0


def test_reason_distribution_counts_majority_reasons():
    assert reason_distribution(A) == {"ok": 1, "casing_error": 1, "meaning_change": 1}


def test_reason_cross_tab_pairs_two_models_reason_codes():
    # Both call p2 reject but for different reasons — the disagreement the
    # binary verdict hides, and the thing effort was shown to move.
    tab = reason_cross_tab(A, B)
    assert tab[("casing_error", "truncation_worse")] == 1
    assert tab[("ok", "ok")] == 1
    assert tab[("meaning_change", "ok")] == 1


def test_reason_cross_tab_ignores_pairs_only_one_model_judged():
    partial = model_votes("model-c", {"p1": ("approve", "ok")})
    assert sum(reason_cross_tab(A, partial).values()) == 1


def test_agreement_uses_the_majority_across_votes():
    # Multi-vote input must collapse first; comparing raw calls would count a
    # 3-vote model three times against a 1-vote model.
    noisy = [
        make_verdict("p1", "approve", "ok", model_id="model-n", run_index=0),
        make_verdict("p1", "reject", "ok", model_id="model-n", run_index=1),
        make_verdict("p1", "approve", "ok", model_id="model-n", run_index=2),
    ]
    steady = model_votes("model-s", {"p1": ("approve", "ok")})
    assert pairwise_agreement(noisy, steady) == 1.0

import pytest

from judge.schema import ReasonCode, Verdict
from judge.vote import VoteResult, flip_rate, majority, tally_votes


def make_verdict(pair_id, verdict, reason, run_index):
    return Verdict(
        pair_id=pair_id,
        verdict=verdict,
        reason=reason,
        model_id="m",
        prompt_version="v1",
        temperature=None,
        run_index=run_index,
    )


def test_majority_picks_most_common():
    assert majority(["approve", "reject", "approve"]) == "approve"


def test_majority_breaks_ties_by_first_occurrence():
    # With an even N, or 3 distinct reason codes, a tie is possible. Ties must
    # resolve deterministically or the same result file would score differently
    # on each run of score.py.
    assert majority(["reject", "approve"]) == "reject"
    assert majority(["approve", "reject"]) == "approve"


def test_flip_rate_is_public_so_one_implementation_serves_both_callers():
    """`judge/flips.py` reports the same metric at a different aggregation level;
    it consumes this function rather than carrying a second copy."""
    assert flip_rate(["approve", "approve", "approve"]) == 0.0
    assert flip_rate(["approve", "reject"]) == 0.5
    assert flip_rate(["approve", "approve", "approve", "reject"]) == 0.25


def test_flip_rate_counts_against_the_modal_value():
    assert flip_rate(["a", "a", "b", "c"]) == 0.5


def test_flip_rate_rejects_empty_naming_itself():
    """A public function must not report a failure under a helper's name.

    Before this, `flip_rate([])` raised "majority() needs at least one value",
    sending a caller to read a function they never called.
    """
    with pytest.raises(ValueError, match="flip_rate"):
        flip_rate([])


def test_majority_rejects_empty():
    with pytest.raises(ValueError):
        majority([])


def test_tally_votes_returns_majority_verdict_and_flip_rates():
    verdicts = [
        make_verdict("p1", "reject", ReasonCode.CASING_ERROR, 0),
        make_verdict("p1", "reject", ReasonCode.TRUNCATION_WORSE, 1),
        make_verdict("p1", "reject", ReasonCode.CASING_ERROR, 2),
    ]
    (result,) = tally_votes(verdicts)
    assert result == VoteResult(
        pair_id="p1",
        verdict="reject",
        reason=ReasonCode.CASING_ERROR,
        n_votes=3,
        verdict_flip_rate=0.0,
        reason_flip_rate=pytest.approx(1 / 3),
    )


def test_tally_votes_tracks_verdict_and_reason_flips_separately():
    # The probe found the binary verdict stable (reject 9/9) while the reason
    # code moved with reasoning effort. Collapsing both into one number would
    # have hidden exactly the instability we have.
    verdicts = [
        make_verdict("p1", "reject", ReasonCode.CASING_ERROR, 0),
        make_verdict("p1", "approve", ReasonCode.OK, 1),
        make_verdict("p1", "reject", ReasonCode.MEANING_CHANGE, 2),
    ]
    (result,) = tally_votes(verdicts)
    assert result.verdict == "reject"
    assert result.verdict_flip_rate == pytest.approx(1 / 3)
    assert result.reason_flip_rate == pytest.approx(2 / 3)


def test_tally_votes_groups_by_pair_and_preserves_first_seen_order():
    verdicts = [
        make_verdict("p2", "approve", ReasonCode.OK, 0),
        make_verdict("p1", "reject", ReasonCode.CASING_ERROR, 0),
        make_verdict("p2", "approve", ReasonCode.OK, 1),
        make_verdict("p1", "reject", ReasonCode.CASING_ERROR, 1),
    ]
    assert [r.pair_id for r in tally_votes(verdicts)] == ["p2", "p1"]
    assert all(r.n_votes == 2 for r in tally_votes(verdicts))


def test_tally_votes_handles_a_single_vote():
    (result,) = tally_votes([make_verdict("p1", "approve", ReasonCode.OK, 0)])
    assert result.n_votes == 1
    assert result.verdict_flip_rate == 0.0


def test_tally_votes_tolerates_uneven_vote_counts():
    # A backend that errored on run 2 leaves that pair with fewer votes; the
    # run still scores rather than aborting, but n_votes records the shortfall.
    verdicts = [
        make_verdict("p1", "approve", ReasonCode.OK, 0),
        make_verdict("p1", "approve", ReasonCode.OK, 1),
        make_verdict("p2", "approve", ReasonCode.OK, 0),
    ]
    by_id = {r.pair_id: r for r in tally_votes(verdicts)}
    assert by_id["p1"].n_votes == 2
    assert by_id["p2"].n_votes == 1

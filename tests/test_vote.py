import pytest

from judge.schema import ReasonCode, Verdict
from judge.vote import VoteResult, flip_rate, majority, settled_majority, tally_votes


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


# --- no majority is a state, not a value (issue #12) ------------------------


def test_settled_majority_is_none_when_the_top_count_is_shared():
    # A winner needs a count no other value matches. Two values tied at the top
    # means the tie-break invented the answer.
    assert settled_majority(["approve", "approve", "reject"]) == "approve"
    assert settled_majority(["approve", "reject"]) is None
    assert settled_majority(["a", "b", "c"]) is None
    assert settled_majority(["only"]) == "only"


def test_settled_majority_rejects_empty_naming_itself():
    with pytest.raises(ValueError, match="settled_majority"):
        settled_majority([])


def test_tally_marks_a_three_way_reason_split_unsettled():
    # deepseek e10-4fd6ba61ea52: verdict settled 3-0, reason split 1-1-1. The
    # verdict is real; the reason was invented by the tie-break.
    verdicts = [
        make_verdict("p1", "reject", ReasonCode.MEANING_CHANGE, 0),
        make_verdict("p1", "reject", ReasonCode.OK, 1),
        make_verdict("p1", "reject", ReasonCode.OVERCORRECTION, 2),
    ]
    (result,) = tally_votes(verdicts)
    assert result.verdict == "reject"
    assert result.verdict_settled is True
    assert result.reason is None
    assert result.reason_settled is False
    assert result.settled is False


def test_tally_marks_an_even_verdict_split_unsettled():
    # inkling stopped at 582/600 on 429s, leaving pairs with two votes. A 1-1
    # verdict split has no majority either — "verdict is binary so it cannot
    # tie" only holds for a COMPLETE run.
    verdicts = [
        make_verdict("p1", "approve", ReasonCode.OK, 0),
        make_verdict("p1", "reject", ReasonCode.CASING_ERROR, 1),
    ]
    (result,) = tally_votes(verdicts)
    assert result.verdict is None
    assert result.verdict_settled is False
    assert result.reason is None


def test_tally_still_settles_an_ordinary_majority():
    verdicts = [
        make_verdict("p1", "approve", ReasonCode.OK, 0),
        make_verdict("p1", "reject", ReasonCode.OK, 1),
        make_verdict("p1", "approve", ReasonCode.OK, 2),
    ]
    (result,) = tally_votes(verdicts)
    assert result.verdict == "approve"
    assert result.reason is ReasonCode.OK
    assert result.settled is True


def test_flip_rate_does_not_depend_on_which_tied_value_wins():
    # flip_rate still uses the modal value, which is safe: when the top count
    # is shared, every candidate yields the same fraction. So the disagreement
    # signal survives even where the winner is meaningless.
    assert flip_rate(["a", "b", "c"]) == pytest.approx(2 / 3)
    assert flip_rate(["c", "b", "a"]) == pytest.approx(2 / 3)
    assert flip_rate(["approve", "reject"]) == 0.5
    assert flip_rate(["reject", "approve"]) == 0.5


def test_unsettled_pair_still_reports_its_flip_rates():
    # The disagreement is the finding. Losing it would make an unsettled pair
    # look like a missing row rather than a contested one.
    verdicts = [
        make_verdict("p1", "reject", ReasonCode.MEANING_CHANGE, 0),
        make_verdict("p1", "reject", ReasonCode.OK, 1),
        make_verdict("p1", "reject", ReasonCode.OVERCORRECTION, 2),
    ]
    (result,) = tally_votes(verdicts)
    assert result.verdict_flip_rate == 0.0
    assert result.reason_flip_rate == pytest.approx(2 / 3)

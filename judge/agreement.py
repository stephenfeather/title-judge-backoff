"""Cross-model agreement — the metrics that work with no operator rulings.

The E10 pack carries 200 title changes and zero verdicts, so kappa-vs-human is
not computable yet. Model-vs-model agreement is, and it answers a question the
leaderboard cannot: where do the candidate judges actually diverge? Items where
models disagree are the same borderline set that flip rates surface, which
makes them the highest-value rows for an operator to rule first.

Verdict agreement and reason agreement are kept apart on purpose. Two models
can both say "reject" for different reasons — that disagreement is invisible in
the binary rate and is exactly what reasoning effort was measured to move.
"""

from __future__ import annotations

from collections import Counter

from judge.schema import Verdict
from judge.vote import tally_votes


def _settled_verdict_by_pair(verdicts: list[Verdict]) -> dict[str, str]:
    """{pair_id: verdict} for pairs whose votes reached a majority.

    Collapsing first is what makes a 3-vote model comparable to a 1-vote one.

    Pairs with no majority are omitted rather than carried as None. Carrying
    them would be actively wrong here: two models that each failed to decide
    would compare None == None and be scored as AGREEING on a pair neither one
    settled. Omitted, they simply fall out of the intersection, which is the
    same treatment a pair a model never judged already gets.
    """
    return {r.pair_id: r.verdict for r in tally_votes(verdicts) if r.verdict is not None}


def _settled_reason_by_pair(verdicts: list[Verdict]) -> dict[str, str]:
    """{pair_id: reason} for pairs whose reason votes reached a majority."""
    return {r.pair_id: r.reason.value for r in tally_votes(verdicts) if r.reason is not None}


def pairwise_agreement(left: list[Verdict], right: list[Verdict]) -> float | None:
    """Fraction of commonly-judged pairs where the two models' verdicts match.

    Compared over the INTERSECTION: a backend that errored on some pairs should
    not drag another model's agreement down. Returns None when the two never
    judged the same pair, which is a different statement from 0.0 agreement.
    """
    a, b = _settled_verdict_by_pair(left), _settled_verdict_by_pair(right)
    shared = a.keys() & b.keys()
    if not shared:
        return None
    return sum(a[pid] == b[pid] for pid in shared) / len(shared)


def agreement_matrix(by_model: dict[str, list[Verdict]]) -> dict[tuple[str, str], float | None]:
    """Verdict agreement for every ordered model pair, diagonal included."""
    return {
        (left, right): pairwise_agreement(by_model[left], by_model[right])
        for left in by_model
        for right in by_model
    }


def reason_distribution(verdicts: list[Verdict]) -> dict[str, int]:
    """How often this model reaches for each reason code (majority per pair).

    A model that never uses a code, or leans on one, is visible here long
    before any ground truth exists.

    Counts only pairs whose reason votes reached a majority. A three-way split
    is not this model reaching for a code — it is the model failing to pick
    one, and crediting a tie-broken value would overstate whichever code the
    tie-break happened to land on. The scenario report enumerates the excluded
    pairs so the omission is visible rather than silent.
    """
    return dict(Counter(_settled_reason_by_pair(verdicts).values()))


def reason_cross_tab(left: list[Verdict], right: list[Verdict]) -> dict[tuple[str, str], int]:
    """Reason-code cross-tab over commonly-judged pairs.

    Off-diagonal mass is disagreement the binary verdict hides.

    A pair counts only where BOTH models settled on a reason; an unsettled
    side has no code to place in a cell.
    """
    a, b = _settled_reason_by_pair(left), _settled_reason_by_pair(right)
    return dict(Counter((a[pid], b[pid]) for pid in a.keys() & b.keys()))

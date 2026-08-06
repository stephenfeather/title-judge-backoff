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


def _majority_by_pair(verdicts: list[Verdict]) -> dict[str, tuple[str, str]]:
    """{pair_id: (verdict, reason)} after collapsing votes to a majority.

    Collapsing first is what makes a 3-vote model comparable to a 1-vote one.
    """
    return {r.pair_id: (r.verdict, r.reason.value) for r in tally_votes(verdicts)}


def pairwise_agreement(left: list[Verdict], right: list[Verdict]) -> float | None:
    """Fraction of commonly-judged pairs where the two models' verdicts match.

    Compared over the INTERSECTION: a backend that errored on some pairs should
    not drag another model's agreement down. Returns None when the two never
    judged the same pair, which is a different statement from 0.0 agreement.
    """
    a, b = _majority_by_pair(left), _majority_by_pair(right)
    shared = a.keys() & b.keys()
    if not shared:
        return None
    return sum(a[pid][0] == b[pid][0] for pid in shared) / len(shared)


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
    """
    return dict(Counter(reason for _, reason in _majority_by_pair(verdicts).values()))


def reason_cross_tab(left: list[Verdict], right: list[Verdict]) -> dict[tuple[str, str], int]:
    """Reason-code cross-tab over commonly-judged pairs.

    Off-diagonal mass is disagreement the binary verdict hides.
    """
    a, b = _majority_by_pair(left), _majority_by_pair(right)
    return dict(Counter((a[pid][1], b[pid][1]) for pid in a.keys() & b.keys()))

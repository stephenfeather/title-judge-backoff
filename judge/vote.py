"""Majority-of-N voting and per-item flip rates.

Repetition, not temperature, is how this harness buys stability: the GPT-5.x
family rejects temperature above reasoning effort `none` and Anthropic has
deprecated the knob, so a single call per pair is a sample from a distribution
we cannot pin. Judging each pair N times and taking the modal verdict cuts the
effective flip rate from f to 3f²-2f³ at N=3 — roughly 13% down to 5%.

Flip rates are recorded per item and reported, not averaged away: items that
flip are the borderline set where the rubric is ambiguous, and a rising
aggregate flip rate is the earliest signal that a provider changed the model
under us.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from judge.schema import ReasonCode, Verdict


@dataclass(frozen=True)
class VoteResult:
    """The agreed ruling for one pair, plus how much its N votes disagreed."""

    pair_id: str
    verdict: str
    reason: ReasonCode
    n_votes: int
    verdict_flip_rate: float  # fraction of votes differing from the modal verdict
    reason_flip_rate: float  # ditto for the reason code, tracked separately


def majority(values: list):
    """Most common value; ties broken by first occurrence.

    The tie-break must be deterministic — otherwise re-scoring the same result
    file could produce a different leaderboard.
    """
    if not values:
        raise ValueError("majority() needs at least one value")
    counts = Counter(values)
    best = max(counts.values())
    # First-occurrence order settles ties; some value always attains `best`.
    return next(value for value in values if counts[value] == best)


def flip_rate(values: list) -> float:
    """Fraction of values differing from the modal one. 0.0 = unanimous.

    Public because two callers need this exact number at two different
    aggregation levels, and two implementations of one metric would drift:

    - `tally_votes` below applies it within a single pair's N votes from one
      model in one run — "did repeating the call change the answer".
    - `judge/flips.py` applies it across every verdict on a pair, spanning
      models and scenarios, for the operator's ruling card — "did this pair
      change anyone's answer".

    Same arithmetic, different populations. If these ever disagreed, the
    leaderboard and the operator card would describe the same pair differently.
    """
    winner = majority(values)
    return sum(v != winner for v in values) / len(values)


def tally_votes(verdicts: list[Verdict]) -> list[VoteResult]:
    """Collapse N votes per pair into one ruling each, in first-seen order."""
    grouped: dict[str, list[Verdict]] = {}
    for v in verdicts:
        grouped.setdefault(v.pair_id, []).append(v)

    return [
        VoteResult(
            pair_id=pair_id,
            verdict=majority([v.verdict for v in votes]),
            reason=majority([v.reason for v in votes]),
            n_votes=len(votes),
            verdict_flip_rate=flip_rate([v.verdict for v in votes]),
            reason_flip_rate=flip_rate([v.reason for v in votes]),
        )
        for pair_id, votes in grouped.items()
    ]

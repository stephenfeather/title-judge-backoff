"""Per-pair disagreement stats over a sweep's verdicts.

The ruling pass is faster and better calibrated when the operator can see which
rows the models fought over — a pair every model approved needs a glance, a pair
they split 50/50 deserves the operator's full attention.

`flip_rate` is `1 - modal share`: 0.0 when every verdict agrees, 0.5 when a pair
is split evenly. It reads "how often does this pair flip a judge's answer".

All of this is optional context. No results directory, a sweep still in flight,
a file of something other than verdicts — every path degrades to "no data", and
the TUI simply omits the pane. The ruling pass never depends on a sweep.

It does not depend on the sweep's *record shape* either. This module reads four
fields — pair_id, verdict, reason, model_id — ignores every other key, and
deliberately imports nothing from `judge.schema`. The verdict record is the
sweep's to evolve (run index, vote index, effort, whatever follows); a pane that
constructed the full schema record would start dropping every line the moment
that record grew a required field, and would go quietly blank instead of failing
loudly. Quiet is the wrong failure mode for context the operator is trusting.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from judge import vote


@dataclass(frozen=True)
class JudgeVerdict:
    """The four fields this pane reads, and nothing else.

    Deliberately not `judge.schema.Verdict`. The sweep's verdict record is the
    sweep's to evolve — run index, vote index, effort, whatever comes next — and
    if loading constructed the full schema record, one new required field there
    would make every line raise and be dropped. The pane would go quietly blank
    rather than fail loudly, which is the worst way for context to disappear.
    `reason` stays a raw string for the same reason: an unrecognized code must
    not erase a verdict, because the flip rate is about approve/reject.
    """

    pair_id: str
    verdict: str
    reason: str
    model_id: str


@dataclass(frozen=True)
class FlipStats:
    """How one pair fared across every judge that ruled on it."""

    pair_id: str
    n: int
    verdicts: dict[str, int]
    reasons: dict[str, int]
    models: list[str]
    majority: str
    flip_rate: float


class VerdictLike(Protocol):
    """Anything carrying the fields this pane reads — `JudgeVerdict`,
    `schema.Verdict`, or whatever richer record the sweep grows next."""

    @property
    def pair_id(self) -> str: ...

    @property
    def verdict(self) -> str: ...

    @property
    def model_id(self) -> str: ...


def _reason_of(verdict: VerdictLike) -> str:
    """The reason as text, whether it arrived as a ReasonCode or a raw string."""
    reason = getattr(verdict, "reason", "")
    return getattr(reason, "value", reason)


def flip_stats(verdicts: Iterable[VerdictLike]) -> dict[str, FlipStats]:
    """Group verdicts by pair and measure how much the judges disagreed.

    Grouped by `pair_id` alone. Repeat votes from one model (majority-of-N) are
    exactly that — repeats — and each one counts, because the question the pane
    answers is "how often did this pair flip an answer", not "how many distinct
    models were asked".
    """
    grouped: dict[str, list[VerdictLike]] = {}
    for verdict in verdicts:
        grouped.setdefault(verdict.pair_id, []).append(verdict)

    return {pair_id: _stats_for(pair_id, group) for pair_id, group in grouped.items()}


def _stats_for(pair_id: str, group: Sequence[VerdictLike]) -> FlipStats:
    values = [v.verdict for v in group]
    counts = Counter(values)
    n = len(group)
    return FlipStats(
        pair_id=pair_id,
        n=n,
        verdicts=dict(counts),
        reasons=dict(Counter(_reason_of(v) for v in group)),
        models=sorted({v.model_id for v in group}),
        majority=vote.majority(values),
        flip_rate=vote.flip_rate(values),
    )


# --- loading (tolerant I/O) ---------------------------------------------------

_REQUIRED = ("pair_id", "verdict", "reason", "model_id")


def _verdict_from_record(record: dict) -> JudgeVerdict | None:
    """Pull the four fields out of a line, or None if they aren't all there.

    Every other key is ignored, whatever it is.
    """
    if not isinstance(record, dict) or any(key not in record for key in _REQUIRED):
        return None
    return JudgeVerdict(
        pair_id=str(record["pair_id"]),
        verdict=str(record["verdict"]),
        reason=str(record["reason"]),
        model_id=str(record["model_id"]),
    )


def load_results(results_dir: Path | str | None) -> list[JudgeVerdict]:
    """Every verdict under `results_dir`, flat or nested one dir per scenario.

    Unreadable files, non-verdict lines and torn tails are skipped rather than
    raised: this is decoration on the ruling card, never a gate on it.
    """
    if results_dir is None:
        return []
    root = Path(results_dir)
    if not root.is_dir():
        return []

    found: list[JudgeVerdict] = []
    for path in sorted(root.rglob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            verdict = _verdict_from_record(record)
            if verdict is not None:
                found.append(verdict)
    return found


def load_flip_stats(results_dir: Path | str | None) -> dict[str, FlipStats]:
    """The single call the TUI makes: results directory -> per-pair stats."""
    return flip_stats(load_results(results_dir))


# --- rendering ----------------------------------------------------------------


def render_flip_pane(stats: FlipStats | None) -> str:
    """One line of sweep context, or nothing at all when no sweep has run."""
    if stats is None:
        return ""
    split = "  ".join(f"{verdict} {count}" for verdict, count in sorted(stats.verdicts.items()))
    reasons = "  ".join(f"{reason} {count}" for reason, count in sorted(stats.reasons.items()))
    models = len(stats.models)
    return (
        f"  sweep: flip {stats.flip_rate:.0%} over {stats.n} votes "
        f"from {models} model{'' if models == 1 else 's'}   {split}\n"
        f"         reasons: {reasons}"
    )

"""Per-pair disagreement stats over a sweep's verdicts.

The ruling pass is faster and better calibrated when the operator can see which
rows the models fought over — a pair every model approved needs a glance, a pair
they split 50/50 deserves the operator's full attention.

`flip_rate` is `1 - modal share`: 0.0 when every verdict agrees, 0.5 when a pair
is split evenly. It reads "how often does this pair flip a judge's answer".

All of this is optional context. No results directory, a sweep still in flight,
a file of something other than verdicts — every path degrades to "no data", and
the TUI simply omits the pane. The ruling pass never depends on a sweep.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from judge.schema import ReasonCode, Verdict


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


def flip_stats(verdicts: Iterable[Verdict]) -> dict[str, FlipStats]:
    """Group verdicts by pair and measure how much the judges disagreed."""
    grouped: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        grouped.setdefault(verdict.pair_id, []).append(verdict)

    return {pair_id: _stats_for(pair_id, group) for pair_id, group in grouped.items()}


def _stats_for(pair_id: str, group: Sequence[Verdict]) -> FlipStats:
    counts = Counter(v.verdict for v in group)
    majority, modal_count = counts.most_common(1)[0]
    n = len(group)
    return FlipStats(
        pair_id=pair_id,
        n=n,
        verdicts=dict(counts),
        reasons=dict(Counter(v.reason.value for v in group)),
        models=sorted({v.model_id for v in group}),
        majority=majority,
        flip_rate=1.0 - modal_count / n,
    )


# --- loading (tolerant I/O) ---------------------------------------------------

_REQUIRED = ("pair_id", "verdict", "reason", "model_id")


def _verdict_from_record(record: dict) -> Verdict | None:
    """Build a Verdict, or None if the line is anything else."""
    if not isinstance(record, dict) or any(key not in record for key in _REQUIRED):
        return None
    try:
        return Verdict(
            pair_id=record["pair_id"],
            verdict=record["verdict"],
            reason=ReasonCode(record["reason"]),
            model_id=record["model_id"],
            prompt_version=record.get("prompt_version", ""),
            temperature=float(record.get("temperature", 0.0)),
        )
    except (ValueError, TypeError):
        return None


def load_results(results_dir: Path | str | None) -> list[Verdict]:
    """Every verdict under `results_dir`, flat or nested one dir per scenario.

    Unreadable files, non-verdict lines and torn tails are skipped rather than
    raised: this is decoration on the ruling card, never a gate on it.
    """
    if results_dir is None:
        return []
    root = Path(results_dir)
    if not root.is_dir():
        return []

    found: list[Verdict] = []
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
    return (
        f"  sweep: flip {stats.flip_rate:.0%} of {stats.n} judge(s)   {split}\n"
        f"         reasons: {reasons}"
    )

"""Record a single backend that deadlocked inside a pooled consensus (#23).

`judge/flips.py` groups by `pair_id` alone, deliberately: the ruling card answers
"how did the slate split on this pair", and its `UNDECIDED` marker means THE
SLATE deadlocked. That marker's value is that it is rare, so it must not start
firing on pairs where 29 of 33 judgments agree.

But a single backend that reached no majority of its own is real information,
and pooling absorbs it. This module records that separately: it never touches
the card, and it is deliberately a different, quieter finding.

Two properties the pooled view cannot provide, both required by #23:

* **Per backend.** A tie inside one judge is invisible once its votes are mixed
  with every other judge's.
* **Per scenario leg.** The same backend runs in `s1` and in each `s2-*` leg.
  Merging them would report a tie no single run produced. The leg is the
  results directory, which `scenario_report` already scopes one per invocation.

Verdict ties and reason ties are recorded separately because they are not the
same finding: across the 2026-08-06 sweep, 12 pairs tied at some backend level
and only ONE of those was a verdict tie. Collapsing them would drown the rare
case in the common one.

The JSONL record exists to be read later, by something that is not this code —
the harness is moving from titles to other and more complex fields, where
ambiguity (and so deadlocking) should be more common, not less. It therefore
carries an explicit schema version and no assumption about what is being judged.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from judge.schema import Verdict
from judge.vote import settled_majority

#: Bumped when the record shape changes in a way a reader must notice.
RECORD_VERSION = 1

VERDICT = "verdict"
REASON = "reason"


@dataclass(frozen=True)
class BackendDeadlock:
    """One backend, one pair, one axis on which its own votes reached no majority."""

    leg: str
    backend: str
    pair_id: str
    kind: str  # VERDICT or REASON — a verdict tie is the far rarer finding
    n_votes: int
    split: dict[str, int]  # what tied, so an audit need not re-read the results


def _tied(values: list[str]) -> bool:
    """True when repeated votes reached no majority.

    A single vote is not a deadlock — there was nothing to disagree with. Two
    identical votes are not either. Only a genuine tie counts, which is what
    `settled_majority` returning None means.
    """
    return len(values) > 1 and settled_majority(values) is None


def backend_deadlocks(by_model: dict[str, list[Verdict]], *, leg: str) -> list[BackendDeadlock]:
    """Every (backend, pair, axis) where that backend alone failed to settle.

    Ordered by backend then pair then axis so two runs over the same results
    produce identical output and diff cleanly.
    """
    found = []
    for backend, verdicts in sorted(by_model.items()):
        by_pair: dict[str, list[Verdict]] = {}
        for verdict in verdicts:
            by_pair.setdefault(verdict.pair_id, []).append(verdict)
        for pair_id, votes in sorted(by_pair.items()):
            axes = (
                (VERDICT, [v.verdict for v in votes]),
                (REASON, [v.reason.value for v in votes]),
            )
            for kind, values in axes:
                if _tied(values):
                    found.append(
                        BackendDeadlock(
                            leg=leg,
                            backend=backend,
                            pair_id=pair_id,
                            kind=kind,
                            n_votes=len(votes),
                            split=dict(sorted(Counter(values).items())),
                        )
                    )
    return found


def deadlock_records(deadlocks: list[BackendDeadlock]) -> str:
    """The audit trail: one JSON object per line, or an empty string for none.

    JSONL rather than a table so it appends across legs and stays queryable by
    whatever reads it later.
    """
    return "".join(
        json.dumps(
            {
                "record_version": RECORD_VERSION,
                "leg": d.leg,
                "backend": d.backend,
                "pair_id": d.pair_id,
                "kind": d.kind,
                "n_votes": d.n_votes,
                "split": d.split,
            },
            sort_keys=True,
        )
        + "\n"
        for d in deadlocks
    )


def render_deadlock_section(deadlocks: list[BackendDeadlock]) -> list[str]:
    """A quiet section for the scenario report, or nothing at all.

    Deliberately worded so it cannot be mistaken for the card's `UNDECIDED`
    marker. That one means the slate deadlocked; this one means one judge did,
    usually inside a consensus the slate reached comfortably.
    """
    if not deadlocks:
        return []
    verdict_ties = [d for d in deadlocks if d.kind == VERDICT]
    lines = [
        "## Backend-level deadlocks",
        "",
        "Pairs where **one backend** reached no majority of its own. This is not the",
        "card's UNDECIDED marker, which means the whole slate deadlocked — most of",
        "these sit inside a consensus the slate reached comfortably, and pooling",
        "would otherwise absorb them entirely.",
        "",
        f"{len(verdict_ties)} verdict tie(s), {len(deadlocks) - len(verdict_ties)} reason tie(s).",
        "A verdict tie is the rarer and stronger finding.",
        "",
        "| Leg | Backend | Pair | Axis | Votes | Split |",
        "|---|---|---|---|---|---|",
    ]
    for d in deadlocks:
        split = ", ".join(f"{value} {count}" for value, count in d.split.items())
        lines.append(
            f"| {d.leg} | {d.backend} | {d.pair_id} | {d.kind} | {d.n_votes} | {split} |"
        )
    lines.append("")
    return lines

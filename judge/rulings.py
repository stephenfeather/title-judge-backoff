"""Append-only ruling journal for the operator pass over the calibration template.

The operator rules 200 rows one keystroke at a time, so the storage has to
survive a crash (or a closed laptop) mid-pass and resume where it stopped.
Nothing is ever rewritten: every keystroke appends exactly one record and
fsyncs, and an `undo` is itself an appended record that replay folds away.

Record shape (one JSON object per line):

    {"ts": "2026-08-06T11:00:00Z", "session": "abc123", "pair_id": "e10-1f2e…",
     "action": "rule" | "skip" | "undo" | "note",
     "ground_truth": "approve" | "reject" | null,
     "reason": "ok" | "casing_error" | … | null,
     "notes": null | "rubric annotation", "auto": false, "elapsed_ms": 1200}

The journal is operator-local and gitignored — it carries vendor-derived
titles' ids and the operator's reasoning, and neither belongs in this repo.
`to_merge_rulings` projects it down to the three fields
`judge.qa_pack.merge_rulings` wants, so the merge contract (and its
stale/partial guards) is untouched by anything that happens in here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from judge.schema import VALID_VERDICTS, ReasonCode

ACTION_RULE = "rule"
ACTION_SKIP = "skip"
ACTION_UNDO = "undo"
ACTION_NOTE = "note"

#: The verdict a `ok` reason belongs to; every other code is a rejection
#: rationale. Kept as a derivation of ReasonCode rather than a second list, so
#: adding a code to the schema cannot silently drift from the keymap below.
_APPROVE_REASON = ReasonCode.OK


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- reason keys --------------------------------------------------------------


#: Keys the TUI binds to verdict actions (`rule.py`). A reason sharing one of
#: these letters would invite a misfire from muscle memory, so the pool skips
#: them; `tests/test_rule.py` keeps the two lists honest.
_ACTION_KEYS = "arsunq"

#: Single-character keys, in allocation order: digits first (the submenu reads
#: as a numbered list), then the letters not already spoken for. Allocation is
#: positional, so adding a reason code never re-letters the ones before it.
KEY_POOL = "123456789" + "".join(
    letter for letter in "abcdefghijklmnopqrstuvwxyz" if letter not in _ACTION_KEYS
)


def allocate_keys(codes: Sequence) -> dict[str, object]:
    """Assign one keystroke per code, or refuse.

    The key reader takes exactly one character, so a code allocated a two-key
    label like "10" would be unreachable — the operator would press `1`, get
    nothing, and have no way to tell that a reason had gone missing. Outgrowing
    the pool is therefore an error at construction rather than a UI that quietly
    drops options.
    """
    if len(codes) > len(KEY_POOL):
        raise ValueError(
            f"{len(codes)} reason codes exceed the {len(KEY_POOL)} single-keystroke "
            "keys available; widen KEY_POOL or split the menu before adding more"
        )
    return {KEY_POOL[index]: code for index, code in enumerate(codes)}


def reason_keymap() -> dict[str, ReasonCode]:
    """Key -> reason code, generated from the schema enum's own order."""
    return allocate_keys(list(ReasonCode))  # type: ignore[return-value]


def reject_reason_keymap() -> dict[str, ReasonCode]:
    """The submenu offered after `r`: every code except the approval one."""
    codes = [code for code in ReasonCode if code is not _APPROVE_REASON]
    return allocate_keys(codes)  # type: ignore[return-value]


# --- validation (pure) --------------------------------------------------------


def _check_ruling(ground_truth: str, reason: ReasonCode) -> None:
    if ground_truth not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {ground_truth!r}")
    if ground_truth == "approve" and reason is not _APPROVE_REASON:
        raise ValueError(
            f"approve must carry reason {_APPROVE_REASON.value!r}, got {reason.value!r}"
        )
    if ground_truth == "reject" and reason is _APPROVE_REASON:
        raise ValueError("reject must carry a failure reason, not 'ok'")


# --- record builders (pure) ---------------------------------------------------


def _base(pair_id: str | None, action: str, elapsed_ms: int | None) -> dict:
    return {
        "pair_id": pair_id,
        "action": action,
        "ground_truth": None,
        "reason": None,
        "notes": None,
        "auto": False,
        "elapsed_ms": elapsed_ms,
    }


def rule_record(
    pair_id: str,
    ground_truth: str,
    reason: ReasonCode,
    *,
    notes: str | None = None,
    auto: bool = False,
    elapsed_ms: int | None = None,
) -> dict:
    """One verdict. `auto` marks a ruling the TUI pre-offered (unchanged rows)."""
    _check_ruling(ground_truth, reason)
    record = _base(pair_id, ACTION_RULE, elapsed_ms)
    record.update(
        ground_truth=ground_truth,
        reason=reason.value,
        notes=notes,
        auto=auto,
    )
    return record


def skip_record(pair_id: str, *, elapsed_ms: int | None = None) -> dict:
    """Deliberately undecided — re-presented on the next pass."""
    return _base(pair_id, ACTION_SKIP, elapsed_ms)


def undo_record(pair_id: str | None = None) -> dict:
    """Retract the most recent ruling (itself journaled, never erased)."""
    return _base(pair_id, ACTION_UNDO, None)


def note_record(pair_id: str, notes: str) -> dict:
    """A rubric annotation. Carries no verdict — the pair stays unruled."""
    record = _base(pair_id, ACTION_NOTE, None)
    record["notes"] = notes
    return record


# --- I/O edge -----------------------------------------------------------------


def append(path: Path | str, record: dict, session: str | None = None) -> dict:
    """Append one record and fsync. Returns the record as written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = dict(record)
    written.setdefault("ts", now_iso())
    if session is not None:
        written.setdefault("session", session)
    line = json.dumps(written, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    return written


def read_records(path: Path | str) -> list[dict]:
    """Read every complete record.

    A truncated final line is dropped: that is a crash mid-write, and every
    record before it is intact. Corruption anywhere else raises — that is data
    loss, and swallowing it would silently shrink the calibration set.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict] = []
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if number == len(lines):
                break  # torn tail from a crash
            raise ValueError(f"{path}: line {number} is corrupt ({exc.msg})") from exc
    return records


# --- replay (pure) ------------------------------------------------------------


@dataclass
class JournalState:
    """Effective state after replaying every record in order."""

    rulings: dict[str, dict] = field(default_factory=dict)
    #: ruling order, so undo knows which pair went last
    order: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    #: full per-pair history, so undo can restore the ruling it superseded
    history: dict[str, list[dict]] = field(default_factory=dict)


def replay(records: Iterable[dict]) -> JournalState:
    """Fold records into effective rulings (last write wins; undo pops)."""
    state = JournalState()
    for record in records:
        action = record.get("action")
        if action == ACTION_UNDO:
            _apply_undo(state)
        elif action == ACTION_SKIP:
            pair_id = record.get("pair_id")
            if pair_id is not None and pair_id not in state.skipped:
                state.skipped.append(str(pair_id))
        elif action == ACTION_NOTE:
            _apply_note(state, record)
        elif action == ACTION_RULE:
            _apply_ruling(state, record)
    return state


def _apply_note(state: JournalState, record: dict) -> None:
    pair_id = record.get("pair_id")
    if pair_id is None or not record.get("notes"):
        return
    state.notes[str(pair_id)] = str(record["notes"])


def _apply_ruling(state: JournalState, record: dict) -> None:
    pair_id = record.get("pair_id")
    if pair_id is None:
        return
    pair_id = str(pair_id)
    if pair_id in state.order:
        state.order.remove(pair_id)
    state.order.append(pair_id)
    state.rulings[pair_id] = record
    state.history.setdefault(pair_id, []).append(record)
    if pair_id in state.skipped:
        state.skipped.remove(pair_id)
    if record.get("notes"):
        state.notes[pair_id] = str(record["notes"])


def _apply_undo(state: JournalState) -> None:
    if not state.order:
        return
    pair_id = state.order.pop()
    history = state.history.get(pair_id, [])
    if history:
        history.pop()
    if history:
        state.rulings[pair_id] = history[-1]
        state.order.append(pair_id)
    else:
        state.rulings.pop(pair_id, None)


def ruled_ids(path: Path | str) -> set[str]:
    """Pair ids already ruled. Skipped pairs are NOT ruled — resume re-presents them."""
    return set(replay(read_records(path)).rulings)


def pending_rows(rows: Sequence[dict], journal_path: Path | str) -> list[dict]:
    """The template minus everything already ruled. Skips come back."""
    done = ruled_ids(journal_path)
    return [row for row in rows if row["id"] not in done]


# --- export (pure + a thin writer) --------------------------------------------


def to_merge_rulings(state: JournalState) -> dict[str, dict]:
    """Project the journal onto exactly the shape `merge_rulings` consumes."""
    return {
        pair_id: {
            "id": pair_id,
            "ground_truth": record["ground_truth"],
            "reason": record["reason"],
        }
        for pair_id, record in state.rulings.items()
    }


def _write_jsonl(path: Path | str, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_merge_rulings(state: JournalState, path: Path | str) -> int:
    """Write the merge-ready rulings JSONL. Returns the row count."""
    return _write_jsonl(path, to_merge_rulings(state).values())


def write_notes(state: JournalState, path: Path | str) -> int:
    """Write rubric annotations to a sidecar, kept out of the merge input."""
    records = ({"id": pair_id, "notes": text} for pair_id, text in state.notes.items())
    return _write_jsonl(path, records)


# --- template helpers (pure) --------------------------------------------------


def is_unchanged(row: dict) -> bool:
    """A no-op row: the pipeline changed nothing.

    Judged on the titles themselves rather than the cohort label — the label is
    the pack's claim, the titles are the fact.
    """
    return row.get("original") == row.get("enriched")

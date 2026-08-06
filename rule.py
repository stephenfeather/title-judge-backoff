# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Rule the calibration template one keystroke at a time.

The E10 pack gives 200 before/after title changes and no verdicts. This is the
human step in the middle: it shows one change per screen and takes a single
keystroke for the verdict, writing an append-only journal that survives a crash
and resumes where it stopped.

    # 1. emit the template (adapt_qa_pack.py)
    uv run adapt_qa_pack.py template --pack title-qa-pack.md --out rulings-template.jsonl

    # 2. rule it — re-runnable, resumes automatically
    uv run rule.py run --template rulings-template.jsonl --journal rulings.journal.jsonl

    # 3. export the rulings and build the calibration set
    uv run rule.py export --journal rulings.journal.jsonl --out rulings.jsonl
    uv run adapt_qa_pack.py merge --template rulings-template.jsonl \
        --rulings rulings.jsonl --out pairs.jsonl

Keys
    a    approve (reason `ok`)
    r    reject -> one-key reason submenu, sourced from judge/schema.py
    ⏎    on an unchanged row only: accept the pre-offered approve
    s    skip (stays pending, re-presented next run)
    u    undo the last action of this session
    n    annotate this row (with --notes)
    q    quit and save

With `--results <dir>` each card also shows how a sweep's judges split on that
pair, and rows are presented most-contested-first — attention is the scarce
resource in a 200-row pass. `--order pack` keeps the template's own order.
Without `--results` the pane is absent and the order is the pack's: the ruling
pass never depends on a sweep having run.

Neither the journal nor the exported rulings belong in this repo: they carry
vendor-derived titles and ids, and `*.jsonl` is gitignored.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, TextIO

from judge import flips, rulings
from judge.flips import FlipStats

KEY_APPROVE = "a"
KEY_REJECT = "r"
KEY_SKIP = "s"
KEY_UNDO = "u"
KEY_NOTE = "n"
KEY_QUIT = "q"
KEYS_ENTER = ("\r", "\n")


# --- input plumbing -----------------------------------------------------------


def keys_from_string(keys: str) -> Callable[[], str]:
    """Key reader over a fixed string (tests and scripted sessions)."""
    stream = iter(keys)
    return lambda: next(stream, "")


def lines_from_iterable(lines: Iterable[str]) -> Callable[[], str]:
    """Line reader over a fixed sequence (tests)."""
    stream = iter(lines)
    return lambda: next(stream, "")


def make_key_reader(stdin: TextIO, isatty: bool) -> Callable[[], str]:
    """Real reader: raw single-char on a TTY, char-at-a-time when piped."""
    if not isatty:
        return lambda: stdin.read(1)

    import termios
    import tty

    fd = stdin.fileno()

    def read_key() -> str:
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            return stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    return read_key


# --- rendering (pure) ---------------------------------------------------------


def word_diff(before: str, after: str) -> str:
    """Token-level before/after: `[rewritten]`, `~recased~`, plain untouched.

    Tokens are aligned case-insensitively on purpose. Casing is the largest
    cohort in the pack, and a case-sensitive alignment reports every recased
    title as a total rewrite — which is exactly the judgment the operator is
    here to make, so the diff must not prejudge it.
    """
    if before == after:
        return "  (no change)"

    old, new = before.split(), after.split()
    matcher = difflib.SequenceMatcher(
        None, [token.casefold() for token in old], [token.casefold() for token in new]
    )
    removed: list[str] = []
    added: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for old_token, new_token in zip(old[i1:i2], new[j1:j2]):
                recased = old_token != new_token
                removed.append(f"~{old_token}~" if recased else old_token)
                added.append(f"~{new_token}~" if recased else new_token)
            continue
        if i1 != i2:
            removed.append("[" + " ".join(old[i1:i2]) + "]")
        if j1 != j2:
            added.append("[" + " ".join(new[j1:j2]) + "]")

    return f"  - {' '.join(removed)}\n  + {' '.join(added)}"


def _stage_line(row: dict) -> str:
    stages = row.get("stages") or []
    return "  stages: " + (", ".join(stages) if stages else "no stages fired")


def render_card(
    row: dict, position: int, total: int, stats: FlipStats | None = None
) -> str:
    """The whole card for one row — one screen, no scrolling."""
    header = f"[{position}/{total}]  {row['id']}   cohort: {row.get('cohort', '?')}"
    parts = [
        header,
        "",
        word_diff(row.get("original", ""), row.get("enriched", "")),
        _stage_line(row),
    ]

    pane = flips.render_flip_pane(stats)
    if pane:
        parts.append(pane)

    if rulings.is_unchanged(row):
        parts.append(
            "  unchanged row (the pipeline made no change) — "
            "ENTER accepts approve/ok, or rule it yourself"
        )

    parts.append("")
    parts.append(_KEY_HINT)
    return "\n".join(parts)


_KEY_HINT = "  a approve   r reject   s skip   u undo   n note   q quit"


def render_reason_menu() -> str:
    """The submenu shown after `r`, generated from the schema enum."""
    options = "   ".join(
        f"{key} {code.value}" for key, code in rulings.reject_reason_keymap().items()
    )
    return f"  reject because:  {options}"


@dataclass
class SessionResult:
    """Per-session throughput, printed on quit."""

    total: int = 0
    ruled: int = 0
    approved: int = 0
    rejected: int = 0
    skipped: int = 0
    undos: int = 0
    notes: int = 0
    quit_early: bool = False
    elapsed_s: float = 0.0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def decisions_per_minute(self) -> float:
        if self.ruled == 0 or self.elapsed_s <= 0:
            return 0.0
        return self.ruled * 60.0 / self.elapsed_s


#: below this, a throughput number says more about the clock than the operator
_MIN_TIMED_SECONDS = 1.0


def render_stats(result: SessionResult) -> str:
    lines = [
        "",
        f"  ruled {result.ruled}/{result.total}   "
        f"approved {result.approved}   rejected {result.rejected}   "
        f"skipped {result.skipped}   undos {result.undos}",
    ]
    if result.elapsed_s >= _MIN_TIMED_SECONDS:
        lines.append(
            f"  {result.decisions_per_minute:.1f} decisions/min "
            f"over {result.elapsed_s:.0f}s"
        )
    if result.reasons:
        tally = "   ".join(f"{reason} {count}" for reason, count in sorted(result.reasons.items()))
        lines.append(f"  reasons: {tally}")
    return "\n".join(lines)


# --- ordering (pure) ----------------------------------------------------------

ORDER_FLIP = "flip"
ORDER_PACK = "pack"


def order_rows(
    rows: Sequence[dict],
    stats: dict[str, FlipStats],
    order: str = ORDER_FLIP,
) -> list[dict]:
    """Rows in the order the operator should see them.

    Default is most-contested-first: attention is the scarce resource in a
    200-row pass, and the rows the judges split on are where an operator ruling
    changes the calibration set most. Ties keep pack order, so the sequence is
    deterministic and a re-run presents rows the same way.

    Rows the sweep never judged sort last rather than first: no verdicts is not
    the same as no disagreement, and ranking them as calm would bury rows nobody
    has looked at yet behind rows everyone agreed on. With no sweep data at all,
    this is exactly pack order.
    """
    if order == ORDER_PACK or not stats:
        return list(rows)

    def rank(indexed: tuple[int, dict]) -> tuple[int, float, int]:
        index, row = indexed
        stat = stats.get(row["id"])
        if stat is None:
            return (1, 0.0, index)  # unjudged: last, pack order among themselves
        return (0, -stat.flip_rate, index)

    return [row for _, row in sorted(enumerate(rows), key=rank)]


def describe_order(order: str, stats: dict[str, FlipStats]) -> str:
    if order == ORDER_PACK or not stats:
        return "pack order"
    return "most contested first"


# --- the loop -----------------------------------------------------------------


def run_session(
    rows: Sequence[dict],
    journal_path: Path | str,
    read_key: Callable[[], str],
    out: TextIO,
    *,
    read_line: Callable[[], str] | None = None,
    stats: dict[str, FlipStats] | None = None,
    notes_enabled: bool = False,
    session: str | None = None,
    now: Callable[[], float] = time.monotonic,
) -> SessionResult:
    """Drive the rapid-fire loop until the rows run out, `q`, or EOF.

    The only side effect is the append-only journal: every accepted keystroke
    appends exactly one record before the screen advances, so an interrupted
    session loses nothing that was shown as decided.
    """
    rows = list(rows)
    stats = stats or {}
    session = session or uuid.uuid4().hex[:8]
    result = SessionResult(total=len(rows))
    #: (row index, action) for session-local undo
    history: list[tuple[int, str]] = []
    started = now()

    index = 0
    while index < len(rows):
        row = rows[index]
        out.write(render_card(row, index + 1, len(rows), stats.get(row["id"])) + "\n")

        key = read_key()
        if key == "":
            break
        if key == KEY_QUIT:
            result.quit_early = True
            break

        if key == KEY_UNDO:
            index = _handle_undo(history, journal_path, session, out, result, index)
            continue

        if key == KEY_NOTE:
            _handle_note(row, journal_path, session, out, result, read_line, notes_enabled)
            continue

        action = _handle_decision(key, row, journal_path, session, read_key, out, result)
        if action is None:
            continue  # unrecognized key, or an aborted rejection — re-present
        history.append((index, action))
        index += 1

    result.elapsed_s = max(0.0, now() - started)
    out.write(render_stats(result) + "\n")
    return result


def _handle_decision(
    key: str,
    row: dict,
    journal_path: Path | str,
    session: str,
    read_key: Callable[[], str],
    out: TextIO,
    result: SessionResult,
) -> str | None:
    """Journal one verdict. Returns the action taken, or None to re-present."""
    if key == KEY_APPROVE:
        record = rulings.rule_record(row["id"], "approve", rulings.ReasonCode.OK)
    elif key in KEYS_ENTER and rulings.is_unchanged(row):
        record = rulings.rule_record(row["id"], "approve", rulings.ReasonCode.OK, auto=True)
    elif key == KEY_REJECT:
        reason = _read_reason(out, read_key)
        if reason is None:
            out.write("  (rejection cancelled)\n")
            return None
        record = rulings.rule_record(row["id"], "reject", reason)
    elif key == KEY_SKIP:
        rulings.append(journal_path, rulings.skip_record(row["id"]), session=session)
        result.skipped += 1
        return rulings.ACTION_SKIP
    else:
        return None

    rulings.append(journal_path, record, session=session)
    result.ruled += 1
    if record["ground_truth"] == "approve":
        result.approved += 1
    else:
        result.rejected += 1
    reason_value = record["reason"]
    result.reasons[reason_value] = result.reasons.get(reason_value, 0) + 1
    return rulings.ACTION_RULE


def _read_reason(out: TextIO, read_key: Callable[[], str]):
    """One key for WHY. An unknown key aborts rather than inventing a code."""
    out.write(render_reason_menu() + "\n")
    return rulings.reject_reason_keymap().get(read_key())


def _handle_note(
    row: dict,
    journal_path: Path | str,
    session: str,
    out: TextIO,
    result: SessionResult,
    read_line: Callable[[], str] | None,
    notes_enabled: bool,
) -> None:
    """`n` annotates the row for the prompt-sharpening pass. No verdict."""
    if not notes_enabled or read_line is None:
        return
    out.write("  note> ")
    text = read_line().strip()
    if not text:
        return
    rulings.append(journal_path, rulings.note_record(row["id"], text), session=session)
    result.notes += 1


def _handle_undo(
    history: list[tuple[int, str]],
    journal_path: Path | str,
    session: str,
    out: TextIO,
    result: SessionResult,
    index: int,
) -> int:
    """Step back one row, retracting a ruling if that is what was last done.

    A skip left no ruling to retract, so undoing one only rewinds the cursor —
    journaling an undo there would pop whatever ruling came before it.
    """
    if not history:
        out.write("  (nothing to undo)\n")
        return index

    previous_index, action = history.pop()
    if action == rulings.ACTION_RULE:
        rulings.append(journal_path, rulings.undo_record(), session=session)
        result.ruled = max(0, result.ruled - 1)
    else:
        result.skipped = max(0, result.skipped - 1)
    result.undos += 1
    return previous_index


# --- CLI ----------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path | str, records: Iterable[dict]) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def cmd_run(
    args: argparse.Namespace,
    read_key: Callable[[], str] | None,
    read_line: Callable[[], str] | None,
    out: TextIO,
) -> None:
    rows = read_jsonl(args.template)
    pending = rulings.pending_rows(rows, args.journal)
    if not pending:
        out.write(f"nothing left to rule — all {len(rows)} rows are ruled\n")
        return

    stats = flips.load_flip_stats(args.results)
    pending = order_rows(pending, stats, args.order)
    out.write(
        f"{len(pending)} of {len(rows)} rows pending  "
        f"({describe_order(args.order, stats)}; journal: {args.journal})\n"
    )
    result = run_session(
        pending,
        args.journal,
        read_key=read_key or make_key_reader(sys.stdin, sys.stdin.isatty()),
        read_line=read_line or (lambda: sys.stdin.readline()),
        out=out,
        stats=stats,
        notes_enabled=args.notes,
    )
    remaining = len(rows) - len(rulings.ruled_ids(args.journal))
    out.write(f"  {remaining} row(s) still unruled\n")
    if result.quit_early:
        out.write("  re-run the same command to pick up where you left off\n")


def cmd_export(args: argparse.Namespace, out: TextIO) -> None:
    state = rulings.replay(rulings.read_records(args.journal))
    count = rulings.write_merge_rulings(state, args.out)
    out.write(f"wrote {args.out} ({count} rulings)\n")
    if args.notes_out:
        notes = rulings.write_notes(state, args.notes_out)
        out.write(f"wrote {args.notes_out} ({notes} annotations)\n")
    if state.skipped:
        out.write(f"note: {len(state.skipped)} row(s) were skipped and remain unruled\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="rule the template one keystroke at a time")
    run.add_argument("--template", type=Path, required=True, help="rulings-template.jsonl")
    run.add_argument("--journal", type=Path, required=True, help="append-only rulings journal")
    run.add_argument("--results", type=Path, help="sweep results dir, for flip-rate context")
    run.add_argument("--notes", action="store_true", help="enable per-row rubric annotations")
    run.add_argument(
        "--order",
        choices=(ORDER_FLIP, ORDER_PACK),
        default=ORDER_FLIP,
        help="row order: 'flip' (default) puts the most contested rows first when "
        "--results is supplied; 'pack' keeps the template's own order",
    )

    export = sub.add_parser("export", help="project the journal into merge-ready rulings")
    export.add_argument("--journal", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True, help="rulings.jsonl for merge")
    export.add_argument("--notes-out", type=Path, help="write annotations to this sidecar")

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    read_key: Callable[[], str] | None = None,
    read_line: Callable[[], str] | None = None,
    out: TextIO | None = None,
) -> None:
    args = build_parser().parse_args(argv)
    sink: TextIO = out if out is not None else sys.stdout
    if args.command == "run":
        cmd_run(args, read_key, read_line, sink)
    else:
        cmd_export(args, sink)


if __name__ == "__main__":
    main()

"""Adapter for the E10 operator QA pack (a markdown table) → calibration pairs.

The pack ships the 200 stratified before/after title changes but carries NO
operator verdicts — it is the proposal document, not the ruling. So the flow
is three steps:

    parse_qa_pack   markdown table  -> PackRow
    template_line   PackRow         -> a JSONL row with ground_truth/reason blank
    merge_rulings   template + rulings -> Pair (the calibration set)

Row ids are content-derived, so a regenerated pack keeps ids stable for rows
whose text did not change and previously-collected rulings still apply.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from judge.schema import Pair, ReasonCode

NO_STAGES = "—"  # em dash: the pipeline made no change to this title
_COLUMNS = 5


@dataclass(frozen=True)
class PackRow:
    cohort: str
    source: str
    before: str
    after: str
    stages: list[str]


def _split_row(line: str) -> list[str] | None:
    """Split one markdown table row into cells, or None if it isn't one."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells = [c.strip() for c in stripped[1:-1].split("|")]
    if len(cells) != _COLUMNS:
        return None
    if all(set(c) <= {"-", ":"} and c for c in cells):  # header separator
        return None
    return cells


def parse_qa_pack(text: str) -> list[PackRow]:
    """Extract sample rows from the pack, ignoring prose and the header."""
    rows = []
    for line in text.splitlines():
        cells = _split_row(line)
        if cells is None:
            continue
        cohort, source, before, after, stages = cells
        if cohort.lower() == "cohort":  # header
            continue
        rows.append(
            PackRow(
                cohort=cohort,
                source=source,
                before=before,
                after=after,
                stages=[] if stages == NO_STAGES else [s.strip() for s in stages.split(",") if s.strip()],
            )
        )
    return rows


def row_id(row: PackRow) -> str:
    """Stable id derived from the title change itself, not row position."""
    digest = hashlib.sha256(f"{row.before}\x00{row.after}".encode()).hexdigest()
    return f"e10-{digest[:12]}"


def template_line(row: PackRow, attributes: dict[str, str] | None = None) -> str:
    """One ruling-template JSONL row: everything known, verdict left blank.

    `attributes` optionally supplies brand/mpn joined from another source; the
    pack itself carries neither, and they are omitted when not supplied.
    """
    record = {
        "id": row_id(row),
        "original": row.before,
        "enriched": row.after,
        "cohort": row.cohort,
        "source": row.source,
        "stages": row.stages,
        "ground_truth": None,
        "reason": None,
    }
    for key in ("brand", "mpn"):
        if attributes and attributes.get(key):
            record[key] = attributes[key]
    return json.dumps(record, ensure_ascii=False)


def merge_rulings(template: list[dict], rulings: dict[str, dict]) -> list[Pair]:
    """Combine the template with operator rulings into the calibration set.

    Fails loudly on either gap: rows nobody ruled, or rulings that match no row
    (a stale id from a regenerated pack).
    """
    known = {record["id"] for record in template}
    unknown = sorted(set(rulings) - known)
    if unknown:
        raise ValueError(f"rulings reference {len(unknown)} unknown pair id(s): {', '.join(unknown)}")

    unruled = [record["id"] for record in template if record["id"] not in rulings]
    if unruled:
        raise ValueError(
            f"{len(unruled)} of {len(template)} rows are unruled; "
            f"first few: {', '.join(unruled[:3])}"
        )

    pairs = []
    for record in template:
        ruling = rulings[record["id"]]
        pairs.append(
            Pair(
                id=record["id"],
                original=record["original"],
                enriched=record["enriched"],
                brand=record.get("brand"),
                mpn=record.get("mpn"),
                ground_truth=ruling["ground_truth"],
                reason=ReasonCode(ruling["reason"]),
            )
        )
    return pairs

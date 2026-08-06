"""Journal + replay + export for the operator ruling pass.

Everything here is pure or tmp_path-local: no TTY, no network, synthetic titles.
"""

from __future__ import annotations

import json

import pytest

from judge import rulings
from judge.schema import ReasonCode


def template_row(pair_id: str, cohort: str = "casing") -> dict:
    return {
        "id": pair_id,
        "original": "acme widget 3000 blk",
        "enriched": "Acme Widget 3000, Black",
        "cohort": cohort,
        "source": "synthetic",
        "stages": ["casing"],
        "ground_truth": None,
        "reason": None,
    }


# --- reason keymap ------------------------------------------------------------


def test_reason_keymap_covers_every_schema_code():
    keymap = rulings.reason_keymap()
    assert set(keymap.values()) == set(ReasonCode)


def test_reason_keymap_uses_single_keystrokes():
    assert all(len(key) == 1 for key in rulings.reason_keymap())


def test_reject_keymap_excludes_ok():
    """`ok` is not a rejection rationale — it must not be offered on reject."""
    assert ReasonCode.OK not in rulings.reject_reason_keymap().values()
    assert set(rulings.reject_reason_keymap().values()) == set(ReasonCode) - {ReasonCode.OK}


# --- record builders ----------------------------------------------------------


def test_rule_record_carries_verdict_and_reason():
    record = rulings.rule_record("e10-aaa", "approve", ReasonCode.OK)
    assert record["pair_id"] == "e10-aaa"
    assert record["action"] == rulings.ACTION_RULE
    assert record["ground_truth"] == "approve"
    assert record["reason"] == "ok"
    assert record["notes"] is None
    assert record["auto"] is False


def test_rule_record_rejects_unknown_verdict():
    with pytest.raises(ValueError, match="verdict"):
        rulings.rule_record("e10-aaa", "maybe", ReasonCode.OK)


def test_rule_record_rejects_approve_with_failure_reason():
    """approve+overcorrection is incoherent ground truth; refuse to journal it."""
    with pytest.raises(ValueError, match="approve"):
        rulings.rule_record("e10-aaa", "approve", ReasonCode.OVERCORRECTION)


def test_rule_record_rejects_reject_with_ok_reason():
    with pytest.raises(ValueError, match="reject"):
        rulings.rule_record("e10-aaa", "reject", ReasonCode.OK)


def test_rule_record_keeps_notes_and_auto_flag():
    record = rulings.rule_record(
        "e10-aaa", "approve", ReasonCode.OK, notes="no-op row", auto=True
    )
    assert record["notes"] == "no-op row"
    assert record["auto"] is True


def test_skip_and_undo_records():
    assert rulings.skip_record("e10-aaa")["action"] == rulings.ACTION_SKIP
    assert rulings.undo_record()["action"] == rulings.ACTION_UNDO


def test_note_record_annotates_without_ruling():
    record = rulings.note_record("e10-aaa", "rubric: brand casing edge case")
    assert record["action"] == rulings.ACTION_NOTE
    assert record["notes"] == "rubric: brand casing edge case"


# --- append / read (I/O edge) -------------------------------------------------


def test_append_stamps_timestamp_and_session(tmp_path):
    path = tmp_path / "rulings.journal.jsonl"
    written = rulings.append(path, rulings.skip_record("e10-aaa"), session="s1")
    assert written["session"] == "s1"
    assert written["ts"].endswith("Z")
    assert len(path.read_text().splitlines()) == 1


def test_append_is_append_only(tmp_path):
    path = tmp_path / "rulings.journal.jsonl"
    rulings.append(path, rulings.rule_record("a", "approve", ReasonCode.OK))
    rulings.append(path, rulings.rule_record("b", "reject", ReasonCode.MEANING_CHANGE))
    assert len(rulings.read_records(path)) == 2


def test_read_records_on_missing_file(tmp_path):
    assert rulings.read_records(tmp_path / "nope.jsonl") == []


def test_read_records_drops_torn_final_line(tmp_path):
    """A crash mid-write loses only the decision in flight."""
    path = tmp_path / "rulings.journal.jsonl"
    rulings.append(path, rulings.rule_record("a", "approve", ReasonCode.OK))
    with path.open("a") as fh:
        fh.write('{"action": "rule", "pair_i')
    records = rulings.read_records(path)
    assert len(records) == 1
    assert records[0]["pair_id"] == "a"


def test_read_records_raises_on_interior_corruption(tmp_path):
    """Damage anywhere but the tail is data loss, not an interrupted write."""
    path = tmp_path / "rulings.journal.jsonl"
    path.write_text('{"broken"\n{"action": "skip", "pair_id": "a"}\n')
    with pytest.raises(ValueError, match="line 1"):
        rulings.read_records(path)


# --- replay -------------------------------------------------------------------


def test_replay_collects_rulings():
    state = rulings.replay(
        [
            rulings.rule_record("a", "approve", ReasonCode.OK),
            rulings.rule_record("b", "reject", ReasonCode.CASING_ERROR),
        ]
    )
    assert set(state.rulings) == {"a", "b"}
    assert state.rulings["b"]["reason"] == "casing_error"


def test_replay_last_write_wins():
    state = rulings.replay(
        [
            rulings.rule_record("a", "approve", ReasonCode.OK),
            rulings.rule_record("a", "reject", ReasonCode.TRUNCATION_WORSE),
        ]
    )
    assert state.rulings["a"]["ground_truth"] == "reject"


def test_replay_undo_pops_last_ruling():
    state = rulings.replay(
        [
            rulings.rule_record("a", "approve", ReasonCode.OK),
            rulings.rule_record("b", "reject", ReasonCode.CASING_ERROR),
            rulings.undo_record(),
        ]
    )
    assert set(state.rulings) == {"a"}


def test_replay_undo_restores_prior_ruling_for_that_pair():
    state = rulings.replay(
        [
            rulings.rule_record("a", "approve", ReasonCode.OK),
            rulings.rule_record("a", "reject", ReasonCode.TRUNCATION_WORSE),
            rulings.undo_record(),
        ]
    )
    assert state.rulings["a"]["ground_truth"] == "approve"


def test_replay_undo_on_empty_journal_is_a_noop():
    assert rulings.replay([rulings.undo_record()]).rulings == {}


def test_replay_skip_leaves_pair_unruled():
    state = rulings.replay([rulings.skip_record("a")])
    assert state.rulings == {}
    assert "a" in state.skipped


def test_replay_ruling_clears_an_earlier_skip():
    state = rulings.replay(
        [rulings.skip_record("a"), rulings.rule_record("a", "approve", ReasonCode.OK)]
    )
    assert "a" in state.rulings
    assert "a" not in state.skipped


def test_replay_collects_notes_from_rule_and_note_records():
    state = rulings.replay(
        [
            rulings.rule_record("a", "approve", ReasonCode.OK, notes="first"),
            rulings.note_record("b", "second"),
        ]
    )
    assert state.notes == {"a": "first", "b": "second"}


def test_replay_note_does_not_rule_the_pair():
    state = rulings.replay([rulings.note_record("a", "just a thought")])
    assert state.rulings == {}


def test_ruled_ids_reads_from_disk(tmp_path):
    path = tmp_path / "j.jsonl"
    rulings.append(path, rulings.rule_record("a", "approve", ReasonCode.OK))
    rulings.append(path, rulings.skip_record("b"))
    assert rulings.ruled_ids(path) == {"a"}


# --- resume -------------------------------------------------------------------


def test_pending_rows_drops_ruled_and_keeps_skipped(tmp_path):
    path = tmp_path / "j.jsonl"
    rulings.append(path, rulings.rule_record("a", "approve", ReasonCode.OK))
    rulings.append(path, rulings.skip_record("b"))
    rows = [template_row("a"), template_row("b"), template_row("c")]
    assert [row["id"] for row in rulings.pending_rows(rows, path)] == ["b", "c"]


def test_pending_rows_on_fresh_journal_returns_everything(tmp_path):
    rows = [template_row("a"), template_row("b")]
    pending = rulings.pending_rows(rows, tmp_path / "absent.jsonl")
    assert [row["id"] for row in pending] == ["a", "b"]


# --- export to the merge contract ---------------------------------------------


def test_to_merge_rulings_emits_exactly_the_merge_fields():
    state = rulings.replay([rulings.rule_record("a", "approve", ReasonCode.OK, notes="n")])
    assert rulings.to_merge_rulings(state) == {
        "a": {"id": "a", "ground_truth": "approve", "reason": "ok"}
    }


def test_to_merge_rulings_output_feeds_merge_rulings(tmp_path):
    """The export is consumed by judge.qa_pack.merge_rulings without adaptation."""
    from judge.qa_pack import merge_rulings

    state = rulings.replay(
        [
            rulings.rule_record("a", "approve", ReasonCode.OK),
            rulings.rule_record("b", "reject", ReasonCode.CASING_ERROR),
        ]
    )
    template = [template_row("a"), template_row("b")]
    pairs = merge_rulings(template, rulings.to_merge_rulings(state))
    assert [p.ground_truth for p in pairs] == ["approve", "reject"]
    assert pairs[1].reason is ReasonCode.CASING_ERROR


def test_write_merge_rulings_writes_one_json_object_per_line(tmp_path):
    state = rulings.replay([rulings.rule_record("a", "approve", ReasonCode.OK)])
    out = tmp_path / "rulings.jsonl"
    count = rulings.write_merge_rulings(state, out)
    assert count == 1
    assert json.loads(out.read_text().strip()) == {
        "id": "a",
        "ground_truth": "approve",
        "reason": "ok",
    }


def test_write_notes_sidecar_keeps_annotations_out_of_the_merge_file(tmp_path):
    state = rulings.replay([rulings.rule_record("a", "approve", ReasonCode.OK, notes="why")])
    out = tmp_path / "notes.jsonl"
    rulings.write_notes(state, out)
    assert json.loads(out.read_text().strip()) == {"id": "a", "notes": "why"}


# --- unchanged cohort ---------------------------------------------------------


def test_unchanged_cohort_is_detected_by_identical_titles():
    row = template_row("a", cohort="unchanged")
    row["enriched"] = row["original"]
    assert rulings.is_unchanged(row)


def test_row_is_not_unchanged_when_titles_differ():
    assert not rulings.is_unchanged(template_row("a", cohort="unchanged"))
